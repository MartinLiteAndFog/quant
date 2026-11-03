#include "Cloudgarden.h"
#include "CloudgardenSlateHUD.h"
#include "Engine/Engine.h"
#include "Engine/GameViewportClient.h"
#include "Widgets/SWeakWidget.h"
void ACloudgardenHUD::BeginPlay(){Super::BeginPlay();if(GEngine&&GEngine->GameViewport){NativeHUDWidget=SNew(SCloudgardenSlateHUD).OwnerHUD(this);GEngine->GameViewport->AddViewportWidgetContent(NativeHUDWidget.ToSharedRef(),10);}}
void ACloudgardenHUD::EndPlay(const EEndPlayReason::Type Reason){if(GEngine&&GEngine->GameViewport&&NativeHUDWidget.IsValid())GEngine->GameViewport->RemoveViewportWidgetContent(NativeHUDWidget.ToSharedRef());NativeHUDWidget.Reset();Super::EndPlay(Reason);}
void ACloudgardenHUD::DrawHUD(){Super::DrawHUD();}
