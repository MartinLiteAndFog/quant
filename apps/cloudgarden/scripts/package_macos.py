#!/usr/bin/env python3
"""Build a standalone local macOS app using Apple's installed SDK; no runtime Python/server dependency."""
from pathlib import Path
import os, plistlib, shutil, struct, subprocess, sys
root=Path(__file__).resolve().parents[1]
target=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/Cloudgarden.app')
resources=target/'Contents/Resources'
mac=target/'Contents/MacOS'
resources.mkdir(parents=True,exist_ok=True);mac.mkdir(parents=True,exist_ok=True)
shutil.copytree(root/'dist',resources/'dist',dirs_exist_ok=True)
with (target/'Contents/Info.plist').open('wb') as f:
    plistlib.dump(dict(CFBundleName='Cloudgarden',CFBundleDisplayName='Cloudgarden',CFBundleIdentifier='com.cloudgarden.local',CFBundleExecutable='Cloudgarden',CFBundlePackageType='APPL',CFBundleShortVersionString='0.2.0',CFBundleVersion='2',CFBundleIconFile='Cloudgarden.icns',NSHighResolutionCapable=True,LSMinimumSystemVersion='13.0',NSAppTransportSecurity={'NSAllowsLocalNetworking':True}),f)
env=dict(os.environ,DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',CLANG_MODULE_CACHE_PATH='/tmp/cloudgarden-clang-cache',SWIFT_MODULECACHE_PATH='/tmp/cloudgarden-swift-cache')
subprocess.run(['xcrun','swiftc',str(root/'scripts/CloudgardenLauncher.swift'),'-o',str(mac/'Cloudgarden'),'-framework','AppKit','-framework','WebKit','-framework','Network','-module-cache-path','/tmp/cloudgarden-swift-cache'],env=env,check=True)
# Icon generation uses AppKit vector paths at each native icon resolution.
iconset=Path('/tmp/cloudgarden-app.iconset');iconset.mkdir(exist_ok=True)
script=Path('/tmp/cloudgarden-icon.swift')
script.write_text('''import AppKit
let folder=CommandLine.arguments[1]
for size in [16,32,64,128,256,512,1024] {
 let image=NSImage(size:NSSize(width:size,height:size));image.lockFocus()
 let scale=CGFloat(size)/512
 let transform=NSAffineTransform();transform.scale(by:scale);transform.concat()
 NSColor(calibratedRed:0.28,green:0.39,blue:0.27,alpha:1).setFill()
 NSBezierPath(roundedRect:NSRect(x:16,y:16,width:480,height:480),xRadius:108,yRadius:108).fill()
 let text="C" as NSString
 text.draw(at:NSPoint(x:110,y:70),withAttributes:[.font:NSFont(name:"Palatino-Bold",size:330) ?? NSFont.boldSystemFont(ofSize:330),.foregroundColor:NSColor(calibratedRed:0.96,green:0.97,blue:0.9,alpha:1)])
 NSColor(calibratedRed:0.81,green:0.68,blue:0.39,alpha:1).setFill()
 let leaf=NSBezierPath();leaf.move(to:NSPoint(x:342,y:369));leaf.curve(to:NSPoint(x:426,y:445),controlPoint1:NSPoint(x:330,y:426),controlPoint2:NSPoint(x:383,y:452));leaf.curve(to:NSPoint(x:342,y:369),controlPoint1:NSPoint(x:438,y:387),controlPoint2:NSPoint(x:387,y:359));leaf.fill()
 image.unlockFocus()
 let bitmap=NSBitmapImageRep(data:image.tiffRepresentation!)!
 try! bitmap.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:"\\(folder)/icon_\\(size).png"))
}
''')
subprocess.run(['xcrun','swift',str(script),str(iconset)],env=env,check=True)
for size in [16,32,64,128,256,512,1024]:
    subprocess.run(['sips','-z',str(size),str(size),str(iconset/f'icon_{size}.png')],check=True,stdout=subprocess.DEVNULL)
for size in [16,32,128,256,512]:
    shutil.copy(iconset/f'icon_{size}.png',iconset/f'icon_{size}x{size}.png')
    shutil.copy(iconset/f'icon_{size*2}.png',iconset/f'icon_{size}x{size}@2x.png')
for file in list(iconset.glob('icon_*.png')):
    if 'x' not in file.stem:file.unlink()
chunks=[]
for kind,name in [('icp4','icon_16x16.png'),('icp5','icon_32x32.png'),('icp6','icon_32x32@2x.png'),('ic07','icon_128x128.png'),('ic08','icon_256x256.png'),('ic09','icon_512x512.png'),('ic10','icon_512x512@2x.png')]:
    payload=(iconset/name).read_bytes();chunks.append(kind.encode()+struct.pack('>I',len(payload)+8)+payload)
payload=b''.join(chunks)
(resources/'Cloudgarden.icns').write_bytes(b'icns'+struct.pack('>I',len(payload)+8)+payload)
subprocess.run(['codesign','--force','--deep','--sign','-',str(target)],check=True)
print(target)
