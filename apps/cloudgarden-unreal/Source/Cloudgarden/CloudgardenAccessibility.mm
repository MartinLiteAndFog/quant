#import <Foundation/NSObject.h>
// Avoid the AppKit umbrella's legacy Carbon FVector declaration colliding with Unreal's FVector.
@interface NSWorkspace : NSObject
+ (NSWorkspace*)sharedWorkspace;
- (BOOL)accessibilityDisplayShouldReduceMotion;
@end
bool CloudgardenReducedMotion(){ @autoreleasepool { return [[NSWorkspace sharedWorkspace] accessibilityDisplayShouldReduceMotion]; } }
