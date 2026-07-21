#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .invoke_handler(tauri::generate_handler![probe_url])
        .run(tauri::generate_context!())
        .expect("error while running fleet desktop");
}

/// Native HTTP probe for the desktop connection panel (bypasses webview CORS).
#[tauri::command]
async fn probe_url(url: String) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(12))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(&url)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status().as_u16();
    let body = resp.text().await.map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_str(&body).unwrap_or_else(|_| {
        serde_json::json!({ "raw": body.chars().take(240).collect::<String>() })
    });
    Ok(serde_json::json!({
        "ok": (200..300).contains(&status),
        "status": status,
        "body": json,
    }))
}
