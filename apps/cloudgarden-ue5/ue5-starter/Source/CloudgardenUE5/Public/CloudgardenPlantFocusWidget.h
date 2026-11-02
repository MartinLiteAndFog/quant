#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "CloudgardenPlantFocusWidget.generated.h"

class UImage;
class UTextureRenderTarget2D;

/** One plant view, never a second canonical organism or a second account. */
USTRUCT(BlueprintType)
struct FCloudgardenPlantEquityPoint
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    int32 DayIndex = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString EquityUsdt;
};

USTRUCT(BlueprintType)
struct FCloudgardenPlantFocusData
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString PlantId;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString PlantName;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString LifeStatus;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString FinancialSource;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString IndividualEquityUsdt;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString NetPnlUsdt;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString HighWaterMarkUsdt;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString MaxDrawdownPct;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString WinRatePct;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    int32 AgeDays = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    int32 Generation = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    int32 MatchingTileCopies = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    TArray<FCloudgardenPlantEquityPoint> EquityPoints;
};

/**
 * Read-only UMG focus drawer. The local Blueprint will pair it with an isolated
 * SceneCaptureComponent2D + RT_CloudgardenPlantFocus in an editor preview stage.
 * Do NOT duplicate a canonical plant Actor or enable seed emitters/physics.
 */
UCLASS(Abstract, Blueprintable)
class CLOUDGARDENUE5_API UCloudgardenPlantFocusWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    /** Only synthetic HUD JSON from the known local bootstrap is allowed here. */
    UFUNCTION(BlueprintCallable, Category="Cloudgarden|Focus")
    bool ImportSyntheticHUDJson(const FString& HudJson);

    /** Called when a plant Actor, tile representation, or rank row is clicked. */
    UFUNCTION(BlueprintCallable, Category="Cloudgarden|Focus")
    bool FocusPlantById(const FString& CanonicalPlantId);

    /** Close the drawer; selected source ID remains available to Atlas/Explore. */
    UFUNCTION(BlueprintCallable, Category="Cloudgarden|Focus")
    void CloseFocusDrawer();

    /** Render target from an OFFSCREEN cosmetic visual-mesh-only SceneCapture. */
    UFUNCTION(BlueprintCallable, Category="Cloudgarden|Focus")
    void SetPlantPreviewRenderTarget(UTextureRenderTarget2D* RenderTarget);

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FCloudgardenPlantFocusData SelectedPlant;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    bool bDrawerOpen = false;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|Focus")
    FString SelectedSourcePlantId;

    UFUNCTION(BlueprintImplementableEvent, Category="Cloudgarden|Focus")
    void OnFocusChanged();

    /** Cosmetic world camera dolly/side framing only: no new scene or day. */
    UFUNCTION(BlueprintImplementableEvent, Category="Cloudgarden|Focus")
    void RequestSoftWorldCameraFocus(const FString& CanonicalPlantId,
                                     float BlendSeconds, float DollyFactor);

    UFUNCTION(BlueprintImplementableEvent, Category="Cloudgarden|Focus")
    void RequestRestoreWorldCamera();

protected:
    UPROPERTY(meta=(BindWidgetOptional), BlueprintReadOnly, Category="Cloudgarden|Focus")
    TObjectPtr<UImage> PlantPreviewImage;

private:
    TMap<FString, FCloudgardenPlantFocusData> SourcePlantRows;
};
