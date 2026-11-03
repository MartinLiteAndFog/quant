#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "CloudgardenHUDWidget.generated.h"

// Data-only Unreal Widget starter. No account/OMS/market API and no UMG assets yet.
// The future Blueprint WBP_CloudgardenHUD binds to these read-only properties.

UENUM(BlueprintType)
enum class ECloudgardenRankingTab : uint8
{
    Winners UMETA(DisplayName="Top 5 Gewinner"),
    Losers UMETA(DisplayName="Top 5 Verlierer")
};

USTRUCT(BlueprintType)
struct FCloudgardenHUDRankRow
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString PlantId;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString PlantName;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString PlantStatus;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString NetPnlUsdt;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString ReturnPct;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    int32 ClosedTrades = 0;
};

USTRUCT(BlueprintType)
struct FCloudgardenHUDTileRow
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString TileId;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString Label;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString CapitalMode;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString PaperBudgetRequestedUsdt;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    int32 VisiblePlantInstances = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    bool bEvolutionEnabled = false;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    bool bPaperBudgetIsFunded = false;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    bool bLiveEnabled = false;
};

UCLASS(Abstract, Blueprintable)
class CLOUDGARDENUE5_API UCloudgardenHUDWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    // Caller must supply a previously validated synthetic HUD snapshot JSON.
    // This class never writes a source genome, finance, tile mode or market order.
    UFUNCTION(BlueprintCallable, Category="Cloudgarden|HUD")
    bool ApplySnapshotJson(const FString& SnapshotJson);

    UFUNCTION(BlueprintCallable, Category="Cloudgarden|HUD")
    void SetRankingTab(ECloudgardenRankingTab NewTab);

    UFUNCTION(BlueprintCallable, Category="Cloudgarden|HUD")
    void SetExploreCompact(bool bCompact);

    UFUNCTION(BlueprintCallable, Category="Cloudgarden|HUD")
    void SetSelectedTile(const FString& TileId);

    UFUNCTION(BlueprintCallable, Category="Cloudgarden|HUD")
    void SetSelectedPlant(const FString& PlantId);

    UFUNCTION(BlueprintPure, Category="Cloudgarden|HUD")
    TArray<FCloudgardenHUDRankRow> GetVisibleTopFive() const;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FText SourceBanner;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FText PortfolioNavText;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FText PortfolioReturnText;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FText NetPnlText;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FText MaxDrawdownText;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FText WinRateText;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString FinancialSourceStatus;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    int32 CanonicalPlantCount = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    int32 ActivePlantCount = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    int32 VisiblePlantInstances = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    int32 TilesEvolving = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    int32 SeedReleases = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    int32 SeedLandings = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    int32 ProposedOffspring = 0;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString RequestedPaperBudgetUsdt;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString FundedPaperBudgetUsdt;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString SelectedTileId;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    FString SelectedPlantId;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    TArray<FCloudgardenHUDTileRow> TileRows;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    ECloudgardenRankingTab RankingTab = ECloudgardenRankingTab::Winners;

    UPROPERTY(BlueprintReadOnly, Category="Cloudgarden|HUD")
    bool bExploreCompact = false;

    UFUNCTION(BlueprintImplementableEvent, Category="Cloudgarden|HUD")
    void OnHUDDataChanged();

    UFUNCTION(BlueprintImplementableEvent, Category="Cloudgarden|HUD")
    void OnRankingTabChanged();

private:
    TArray<FCloudgardenHUDRankRow> WinnerRows;
    TArray<FCloudgardenHUDRankRow> LoserRows;
};
