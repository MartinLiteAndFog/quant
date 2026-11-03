#pragma once

#include "CoreMinimal.h"
#include "Widgets/SLeafWidget.h"

/** A bounded, unsmoothed series of measured capital values. X is simulation day. */
class SCloudgardenCurve : public SLeafWidget
{
public:
    SLATE_BEGIN_ARGS(SCloudgardenCurve)
        : _Unit(TEXT("USDT")), _ShowAxes(true) {}
        SLATE_ARGUMENT(TArray<FVector2D>, Points)
        SLATE_ARGUMENT(FString, Unit)
        SLATE_ARGUMENT(bool, ShowAxes)
    SLATE_END_ARGS()

    void Construct(const FArguments& Args);
    void SetPoints(const TArray<FVector2D>& InPoints);

protected:
    virtual FVector2D ComputeDesiredSize(float) const override;
    virtual int32 OnPaint(const FPaintArgs&, const FGeometry&, const FSlateRect&,
        FSlateWindowElementList&, int32, const FWidgetStyle&, bool) const override;

private:
    TArray<FVector2D> Points;
    FString Unit;
    bool ShowAxes = true;
};
