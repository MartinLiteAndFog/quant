import AppKit
import WebKit
import Network

// Local-only static server. The application never contacts an external host.
final class GardenApp: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    var window: NSWindow!
    var listener: NWListener!
    let queue = DispatchQueue(label: "cloudgarden.loopback")
    let root = Bundle.main.resourceURL!.appendingPathComponent("dist").standardizedFileURL
    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            let parameters = NWParameters.tcp
            parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: 5188)
            listener = try NWListener(using: parameters)
            listener.newConnectionHandler = { [weak self] connection in self?.serve(connection) }
            listener.stateUpdateHandler = { [weak self] state in
                if case .ready = state { DispatchQueue.main.async { self?.showGarden() } }
                if case .failed(let error) = state { DispatchQueue.main.async { self?.fail(error.localizedDescription) } }
            }
            listener.start(queue: queue)
        } catch { fail(error.localizedDescription) }
    }
    func showGarden() {
        let view = WKWebView(frame: .zero)
        view.navigationDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 850), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "Cloudgarden"
        window.contentView = view
        window.center(); window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        view.load(URLRequest(url: URL(string: "http://127.0.0.1:5188/")!))
    }
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        let url = navigationAction.request.url
        decisionHandler(url?.host == "127.0.0.1" || url?.scheme == "blob" ? .allow : .cancel)
    }
    func serve(_ connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16384) { [weak self] data, _, _, _ in
            guard let self = self, let data = data, let request = String(data: data, encoding: .utf8), let line = request.components(separatedBy: "\r\n").first else { connection.cancel(); return }
            let parts = line.split(separator: " ")
            guard parts.count >= 2, parts[0] == "GET" || parts[0] == "HEAD" else { connection.cancel(); return }
            let path = String(parts[1]).components(separatedBy: "?")[0].removingPercentEncoding ?? ""
            let file = self.root.appendingPathComponent(path == "/" ? "index.html" : String(path.dropFirst())).standardizedFileURL
            let safe = file.path.hasPrefix(self.root.path + "/")
            let body = safe ? try? Data(contentsOf: file) : nil
            let mime = ["html":"text/html; charset=utf-8", "js":"text/javascript", "css":"text/css", "json":"application/json", "svg":"image/svg+xml", "png":"image/png", "woff2":"font/woff2"][file.pathExtension] ?? "application/octet-stream"
            let payload = body ?? Data("Not found".utf8)
            var response = Data("HTTP/1.1 \(body == nil ? "404 Not Found" : "200 OK")\r\nContent-Type: \(mime)\r\nContent-Length: \(payload.count)\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Security-Policy: default-src 'self' blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'\r\n\r\n".utf8)
            if parts[0] != "HEAD" { response.append(payload) }
            connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
        }
    }
    func fail(_ message: String) { let alert = NSAlert(); alert.messageText = "Cloudgarden konnte nicht starten"; alert.informativeText = message + "\nSchließe eine bereits laufende Cloudgarden-App und versuche es erneut."; alert.runModal(); NSApp.terminate(nil) }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) { listener?.cancel() }
}
let app = NSApplication.shared
let delegate = GardenApp()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
