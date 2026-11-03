#pragma once
#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
class ACloudgardenHUD;
class ACloudgardenController;
class ACloudgardenMode;
class SCloudgardenSlateHUD : public SCompoundWidget {
public:
 SLATE_BEGIN_ARGS(SCloudgardenSlateHUD){} SLATE_ARGUMENT(TWeakObjectPtr<ACloudgardenHUD>,OwnerHUD) SLATE_END_ARGS()
 void Construct(const FArguments& Args);
 virtual void Tick(const FGeometry& Geometry,double Time,float Delta) override;
 virtual bool SupportsKeyboardFocus() const override {return false;}
private:
 TWeakObjectPtr<ACloudgardenHUD> Owner;
 FString LastKey;
 float LastWidth=0;
 TSharedPtr<SWidget> DockPanel;
 TSharedPtr<SWidget> Portrait;
 int DrawerTab=0;
 FVector2D CalloutPosition() const;
 TSharedRef<SWidget> Build(float Width);
 FReply Act(const FString& Action);
 ACloudgardenController* Controller() const;
 ACloudgardenMode* Mode() const;
};
