#pragma once

#include "CoreMinimal.h"
#include "Widgets/SLeafWidget.h"

/** Continuous neutral edge shading, not separate card backgrounds. */
class SCloudgardenHUDScrim : public SLeafWidget
{
public:
    SLATE_BEGIN_ARGS(SCloudgardenHUDScrim) {} SLATE_END_ARGS()
    void Construct(const FArguments&);
protected:
    virtual FVector2D ComputeDesiredSize(float) const override;
    virtual int32 OnPaint(const FPaintArgs&, const FGeometry&, const FSlateRect&,
        FSlateWindowElementList&, int32, const FWidgetStyle&, bool) const override;
};
