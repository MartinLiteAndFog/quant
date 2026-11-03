#pragma once

#include "CoreMinimal.h"
#include "Widgets/SLeafWidget.h"
#include "Styling/SlateBrush.h"
#include "UObject/StrongObjectPtr.h"
#include "Engine/TextureRenderTarget2D.h"

class ACloudgardenHUD;
class ACloudgardenPlant;
class ASceneCapture2D;
class UTextureRenderTarget2D;
class FJsonObject;

/** A transparent live portrait of the selected plant, captured only on state changes. */
class SCloudgardenPlantPortrait : public SLeafWidget
{
public:
    SLATE_BEGIN_ARGS(SCloudgardenPlantPortrait) : _OwnerHUD(nullptr) {}
        SLATE_ARGUMENT(ACloudgardenHUD*, OwnerHUD)
    SLATE_END_ARGS()
    void Construct(const FArguments&);
    virtual ~SCloudgardenPlantPortrait() override;
    virtual void Tick(const FGeometry&, double, float) override;
protected:
    virtual FVector2D ComputeDesiredSize(float) const override;
    virtual int32 OnPaint(const FPaintArgs&, const FGeometry&, const FSlateRect&,
        FSlateWindowElementList&, int32, const FWidgetStyle&, bool) const override;
private:
    TWeakObjectPtr<ACloudgardenHUD> Owner;
    TWeakObjectPtr<ACloudgardenPlant> LastPlant;
    TWeakPtr<FJsonObject> LastData;
    TWeakObjectPtr<ASceneCapture2D> Capture;
    TStrongObjectPtr<UTextureRenderTarget2D> Target;
    FSlateBrush Brush;
    bool HasPortrait = false;
};
