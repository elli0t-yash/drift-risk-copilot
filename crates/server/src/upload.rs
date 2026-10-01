//! `POST /portfolio/upload`: turns a CSV or XLSX file into a `Portfolio`
//! the caller can hand straight to `/experiment` or `/ask`.
//!
//! Column detection recognises the header conventions of several major
//! Indian brokers' holdings exports (Zerodha, Upstox, Groww, Angel One,
//! HDFC Securities, ICICI Direct), not just this codebase's own canonical
//! `ticker`/`weight`/`shares`/`avg_price_inr` names -- see
//! `TICKER_ALIASES`/`QUANTITY_ALIASES`/`PRICE_ALIASES`/`WEIGHT_ALIASES`.

use std::collections::BTreeMap;
use std::io::Cursor;

use axum::extract::Multipart;
use calamine::{open_workbook_from_rs, Data, DataType, Reader, Xlsx};
use compute::experiments::{Holding, Portfolio};
use serde::Serialize;

use crate::error::ApiError;
use crate::validate::validate_portfolio;

/// Weight-based CSV/XLSX uploads carry no portfolio value at all (only
/// ticker + weight), so there is nothing to derive `total_value_inr` from.
/// Defaults to this codebase's existing demo convention (the same value
/// `sample_portfolio` helpers across the test suite use) — callers that
/// care about the real value should overwrite it in the returned
/// `Portfolio` before using it.
const DEFAULT_TOTAL_VALUE_INR: f64 = 1_000_000.0;

/// Header names (lowercase, matching how `parse_csv`/`parse_xlsx` store
/// them) a ticker column might be called across the brokers this endpoint
/// supports. `isin`/`script` are deliberately included even though an
/// ISIN isn't actually a trading symbol -- a few export formats use the
/// column for the symbol anyway, and there's no other candidate column to
/// fall back to in those files.
const TICKER_ALIASES: &[&str] = &[
    "ticker",
    "symbol",
    "instrument",
    "stock",
    "trading_symbol",
    "scrip",
    "isin",
    "script",
    "stock name",
    "company name",
    "name",
];

const QUANTITY_ALIASES: &[&str] =
    &["qty", "qty.", "quantity", "shares", "units", "net quantity", "holdings", "volume", "net qty", "net qty."];

/// Includes `ltp`/`last price`: some exports carry only the current market
/// price, not the original purchase price. Using it still produces a
/// internally-consistent *current* weight split (qty * price / total) --
/// it just means the derived `total_value_inr` reflects today's value
/// rather than cost basis, which is arguably the more useful number for a
/// risk tool anyway.
const PRICE_ALIASES: &[&str] = &[
    "avg_price_inr",
    "avg. cost",
    "avg cost",
    "average_price",
    "avg price",
    "avg. price",
    "avg rate",
    "average rate",
    "average buy price",
    "avg. buy price",
    "avg buy price",
    "buy avg",
    "avg. buy rate",
    "cost price",
    "purchase price",
    "ltp",
    "last price",
];

const WEIGHT_ALIASES: &[&str] = &["weight", "weight%", "weight %", "allocation", "allocation%", "%"];

/// How many leading rows `parse_xlsx` scans for the real header row before
/// giving up and treating row 0 as the header (see its doc comment).
const MAX_HEADER_SCAN_ROWS: usize = 20;

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub portfolio: Portfolio,
    pub layout_detected: &'static str,
    /// `"RELIANCE" -> "RELIANCE.NS"` for every ticker that got normalised
    /// (suffix appended and/or exchange prefix stripped); empty if none
    /// needed it.
    pub tickers_normalised: Vec<String>,
    pub row_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Layout {
    Weight,
    Value,
}

type Record = BTreeMap<String, String>;

/// Resolved column names (as they actually appear in this file's header
/// row) for each role the parser understands, plus which `Layout` that
/// implies. `quantity`/`price` are `None` under `Layout::Weight` and vice
/// versa for `weight` under `Layout::Value`.
struct ColumnMap {
    ticker: String,
    quantity: Option<String>,
    price: Option<String>,
    weight: Option<String>,
    layout: Layout,
}

pub async fn post_portfolio_upload(mut multipart: Multipart) -> Result<axum::Json<UploadResponse>, ApiError> {
    let mut filename: Option<String> = None;
    let mut bytes: Option<Vec<u8>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request("invalid_upload", format!("malformed multipart body: {e}")))?
    {
        if field.name() == Some("file") {
            filename = field.file_name().map(|s| s.to_string());
            let data = field
                .bytes()
                .await
                .map_err(|e| ApiError::bad_request("invalid_upload", format!("failed to read file bytes: {e}")))?;
            bytes = Some(data.to_vec());
        }
    }

    let filename =
        filename.ok_or_else(|| ApiError::bad_request("invalid_upload", "missing a \"file\" form field"))?;
    let bytes = bytes.ok_or_else(|| ApiError::bad_request("invalid_upload", "missing a \"file\" form field"))?;

    if bytes.is_empty() {
        return Err(ApiError::bad_request("empty_file", "uploaded file is empty"));
    }

    let lower_name = filename.to_lowercase();
    let records = if lower_name.ends_with(".csv") {
        parse_csv(&bytes)?
    } else if lower_name.ends_with(".xlsx") {
        parse_xlsx(&bytes)?
    } else {
        return Err(ApiError::bad_request(
            "unsupported_file_type",
            format!("unsupported file type for \"{filename}\"; expected .csv or .xlsx"),
        ));
    };

    if records.is_empty() {
        return Err(ApiError::bad_request("empty_file", "file contains no data rows"));
    }

    let headers: Vec<String> = records[0].keys().cloned().collect();
    let columns = resolve_columns(&headers)?;
    let (holdings, tickers_normalised, total_value_inr) = match columns.layout {
        Layout::Weight => build_weight_based(&records, &columns)?,
        Layout::Value => build_value_based(&records, &columns)?,
    };

    let row_count = holdings.len();
    let portfolio = Portfolio { holdings, total_value_inr };
    validate_portfolio(&portfolio)?;

    Ok(axum::Json(UploadResponse {
        portfolio,
        layout_detected: match columns.layout {
            Layout::Weight => "weight",
            Layout::Value => "value",
        },
        tickers_normalised,
        row_count,
    }))
}

/// Case-insensitive header lookup: `field()` on a `Record` (whose keys are
/// already lowercased in `parse_csv`/`parse_xlsx`).
fn field<'a>(record: &'a Record, name: &str) -> Option<&'a str> {
    record.get(name).map(|s| s.as_str())
}

/// The first header in `headers` matching one of `aliases` (both sides
/// already lowercase/trimmed by the caller).
fn find_column(headers: &[String], aliases: &[&str]) -> Option<String> {
    headers.iter().find(|h| aliases.contains(&h.as_str())).cloned()
}

fn resolve_columns(headers: &[String]) -> Result<ColumnMap, ApiError> {
    let Some(ticker) = find_column(headers, TICKER_ALIASES) else {
        return Err(unsupported_format_error(headers));
    };

    // Weight takes precedence when a file happens to carry both a weight
    // column and quantity+price columns -- it's the more direct signal of
    // intended allocation, not a derived one.
    if let Some(weight) = find_column(headers, WEIGHT_ALIASES) {
        return Ok(ColumnMap { ticker, quantity: None, price: None, weight: Some(weight), layout: Layout::Weight });
    }

    let quantity = find_column(headers, QUANTITY_ALIASES);
    let price = find_column(headers, PRICE_ALIASES);
    if let (Some(quantity), Some(price)) = (quantity, price) {
        return Ok(ColumnMap {
            ticker,
            quantity: Some(quantity),
            price: Some(price),
            weight: None,
            layout: Layout::Value,
        });
    }

    Err(unsupported_format_error(headers))
}

fn unsupported_format_error(headers: &[String]) -> ApiError {
    let columns = headers.join(", ");
    ApiError::bad_request(
        "missing_columns",
        format!(
            "Could not parse portfolio file. Columns found: [{columns}]. Supported brokers: Zerodha, \
             Upstox, Groww, Angel One, HDFC Securities, ICICI Direct. Expected columns: a ticker \
             column (Instrument/Symbol/trading_symbol) and either a weight column or quantity + price \
             columns."
        ),
    )
}

/// Whether a row should be skipped rather than treated as a holding: an
/// empty ticker, a "Total"/"Grand Total" summary row brokers routinely
/// append, or a row whose ticker cell starts with a digit (seen in some
/// exports' footnote/disclaimer rows).
fn should_skip_row(raw_ticker: &str) -> bool {
    let trimmed = raw_ticker.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_lowercase();
    if lower == "total" || lower == "grand total" {
        return true;
    }
    trimmed.chars().next().is_some_and(|c| c.is_ascii_digit())
}

fn parse_number(record: &Record, name: &str, row_index: usize) -> Result<f64, ApiError> {
    let raw = field(record, name)
        .ok_or_else(|| ApiError::bad_request("missing_columns", format!("row {row_index}: missing \"{name}\"")))?;
    raw.trim().parse::<f64>().map_err(|_| {
        ApiError::bad_request("invalid_upload", format!("row {row_index}: \"{name}\" is not a number: {raw:?}"))
    })
}

/// Strips a leading exchange prefix ("NSE:RELIANCE", "BSE:RELIANCE" ->
/// "RELIANCE"), then appends `.NS` to a ticker that has no suffix and
/// looks like a bare NSE symbol (letters/digits only, no `.`) -- both
/// exchanges normalise to the same `.NS` Yahoo Finance suffix this
/// codebase uses everywhere else (see `compute::data`), not `.BO`; the two
/// trade near-identical prices for any name liquid enough to appear in a
/// retail holdings export, and introducing a second suffix convention
/// nothing else in the pipeline understands isn't worth it for that
/// difference. Returns `(normalised_ticker, Some("ORIGINAL -> NORMALISED")
/// if it changed)`.
fn normalise_ticker(raw: &str) -> (String, Option<String>) {
    let trimmed = raw.trim();
    let without_prefix = trimmed.split_once(':').map(|(_, rest)| rest.trim()).unwrap_or(trimmed);

    let looks_bare_nse = !without_prefix.contains('.')
        && !without_prefix.is_empty()
        && without_prefix.chars().all(|c| c.is_ascii_alphanumeric());

    if looks_bare_nse {
        let normalised = format!("{}.NS", without_prefix.to_uppercase());
        (normalised.clone(), Some(format!("{trimmed} -> {normalised}")))
    } else if without_prefix != trimmed {
        (without_prefix.to_string(), Some(format!("{trimmed} -> {without_prefix}")))
    } else {
        (without_prefix.to_string(), None)
    }
}

fn build_weight_based(records: &[Record], columns: &ColumnMap) -> Result<(Vec<Holding>, Vec<String>, f64), ApiError> {
    let weight_col = columns.weight.as_deref().expect("Layout::Weight always has a weight column");
    let mut holdings = Vec::with_capacity(records.len());
    let mut tickers_normalised = Vec::new();

    for (i, record) in records.iter().enumerate() {
        let raw_ticker = field(record, &columns.ticker)
            .ok_or_else(|| ApiError::bad_request("missing_columns", format!("row {i}: missing ticker column")))?;
        if should_skip_row(raw_ticker) {
            continue;
        }
        let (ticker, note) = normalise_ticker(raw_ticker);
        if let Some(note) = note {
            tickers_normalised.push(note);
        }
        let weight = parse_number(record, weight_col, i)?;
        holdings.push(Holding { ticker, weight });
    }

    Ok((holdings, tickers_normalised, DEFAULT_TOTAL_VALUE_INR))
}

fn build_value_based(records: &[Record], columns: &ColumnMap) -> Result<(Vec<Holding>, Vec<String>, f64), ApiError> {
    let quantity_col = columns.quantity.as_deref().expect("Layout::Value always has a quantity column");
    let price_col = columns.price.as_deref().expect("Layout::Value always has a price column");
    let mut rows = Vec::with_capacity(records.len());
    let mut tickers_normalised = Vec::new();

    for (i, record) in records.iter().enumerate() {
        let raw_ticker = field(record, &columns.ticker)
            .ok_or_else(|| ApiError::bad_request("missing_columns", format!("row {i}: missing ticker column")))?;
        if should_skip_row(raw_ticker) {
            continue;
        }
        let (ticker, note) = normalise_ticker(raw_ticker);
        if let Some(note) = note {
            tickers_normalised.push(note);
        }
        let shares = parse_number(record, quantity_col, i)?;
        let avg_price_inr = parse_number(record, price_col, i)?;
        rows.push((ticker, shares * avg_price_inr));
    }

    let total_value_inr: f64 = rows.iter().map(|(_, v)| v).sum();
    let round6 = |x: f64| (x * 1_000_000.0).round() / 1_000_000.0;
    let mut weights: Vec<f64> = rows
        .iter()
        .map(|(_, value_inr)| if total_value_inr > 0.0 { round6(value_inr / total_value_inr) } else { 0.0 })
        .collect();
    // Rounding each weight to 6dp independently can leave the reported
    // weights summing to slightly more/less than 1.0 once enough holdings
    // are involved (the same issue `upstox::holdings_to_portfolio` hit and
    // fixed the same way) -- absorb the residual into the last holding so
    // the sum is exact to float precision rather than left to chance.
    if total_value_inr > 0.0 {
        let rounded_sum: f64 = weights.iter().sum();
        if let Some(last) = weights.last_mut() {
            *last = round6(*last + (1.0 - rounded_sum));
        }
    }
    let holdings: Vec<Holding> =
        rows.into_iter().zip(weights).map(|((ticker, _), weight)| Holding { ticker, weight }).collect();

    Ok((holdings, tickers_normalised, total_value_inr))
}

fn parse_csv(bytes: &[u8]) -> Result<Vec<Record>, ApiError> {
    let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(bytes);
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| ApiError::bad_request("invalid_upload", format!("failed to read CSV headers: {e}")))?
        .iter()
        .map(|h| h.trim().to_lowercase())
        .collect();

    let mut records = Vec::new();
    for result in reader.records() {
        let row = result.map_err(|e| ApiError::bad_request("invalid_upload", format!("malformed CSV row: {e}")))?;
        let mut record = Record::new();
        for (header, value) in headers.iter().zip(row.iter()) {
            record.insert(header.clone(), value.trim().to_string());
        }
        records.push(record);
    }
    Ok(records)
}

/// Scans the first `MAX_HEADER_SCAN_ROWS` rows for the real header row
/// (the one containing a recognised ticker-column name) before falling
/// back to row 0 -- Zerodha's XLSX export (and similar broker exports)
/// prepends metadata rows (account name, date range, disclaimers) above
/// the actual table.
fn parse_xlsx(bytes: &[u8]) -> Result<Vec<Record>, ApiError> {
    let cursor = Cursor::new(bytes.to_vec());
    let mut workbook: Xlsx<_> = open_workbook_from_rs(cursor)
        .map_err(|e| ApiError::bad_request("invalid_upload", format!("failed to open XLSX file: {e}")))?;

    let sheet_name = workbook
        .sheet_names()
        .first()
        .cloned()
        .ok_or_else(|| ApiError::bad_request("invalid_upload", "XLSX file has no sheets"))?;
    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| ApiError::bad_request("invalid_upload", format!("failed to read XLSX sheet: {e}")))?;

    let all_rows: Vec<&[Data]> = range.rows().collect();
    if all_rows.is_empty() {
        return Ok(Vec::new());
    }

    let header_idx = all_rows
        .iter()
        .take(MAX_HEADER_SCAN_ROWS)
        .position(|row| {
            row.iter().any(|cell| TICKER_ALIASES.contains(&cell_to_string(cell).to_lowercase().as_str()))
        })
        .unwrap_or(0);

    let headers: Vec<String> = all_rows[header_idx].iter().map(|cell| cell_to_string(cell).to_lowercase()).collect();

    let mut records = Vec::new();
    for row in &all_rows[header_idx + 1..] {
        let mut record = Record::new();
        for (header, cell) in headers.iter().zip(row.iter()) {
            record.insert(header.clone(), cell_to_string(cell));
        }
        records.push(record);
    }
    Ok(records)
}

fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::String(s) => s.trim().to_string(),
        Data::Float(f) => {
            // Numeric cells (e.g. a "weight" column typed as a number by
            // Excel) round-trip through the same string parsing path as
            // CSV, rather than adding a separate numeric branch everywhere
            // a column is read.
            let mut s = format!("{f}");
            if s.ends_with(".0") {
                s.truncate(s.len() - 2);
            }
            s
        }
        Data::Int(i) => i.to_string(),
        _ => cell.as_string().unwrap_or_default(),
    }
}
