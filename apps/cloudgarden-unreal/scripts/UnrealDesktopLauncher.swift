import AppKit
final class Launcher:NSObject,NSApplicationDelegate {
 var bridge:Process?;var game:Process?
 func applicationDidFinishLaunching(_ notification:Notification){
  do{
   let resources=Bundle.main.resourceURL!
   let data=FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0].appendingPathComponent("Cloudgarden/SolanaFocus")
   try FileManager.default.createDirectory(at:data,withIntermediateDirectories:true)
   let simulation=resources.appendingPathComponent("Simulation")
   let bridge=Process();self.bridge=bridge
   bridge.executableURL=simulation.appendingPathComponent("runtime/bin/node")
   bridge.arguments=[simulation.appendingPathComponent("bridge.mjs").path,data.path]
   var environment=ProcessInfo.processInfo.environment;environment["CLOUDGARDEN_ARCHIVE_ROOT"]=simulation.appendingPathComponent("archives").path;bridge.environment=environment
   try bridge.run()
   let game=Process();self.game=game
   game.executableURL=resources.appendingPathComponent("Game.app/Contents/MacOS/Cloudgarden")
   game.arguments=["-CloudgardenDataDir=\(data.path)","-windowed","-ResX=1440","-ResY=900"]
   game.terminationHandler={_ in bridge.terminate();DispatchQueue.main.async{NSApplication.shared.terminate(nil)}}
   try game.run()
  }catch{bridge?.terminate();let alert=NSAlert();alert.messageText="Cloudgarden konnte nicht starten";alert.informativeText=error.localizedDescription;alert.runModal();NSApplication.shared.terminate(nil)}
 }
 func applicationWillTerminate(_ notification:Notification){bridge?.terminate()}
}
let app=NSApplication.shared;let launcher=Launcher();app.delegate=launcher;app.run()
