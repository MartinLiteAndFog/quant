#include "CloudgardenHUDWidget.h"

#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
    bool TextValue(const TSharedPtr<FJsonObject>& Object, const TCHAR* Key, FString& Value)
    {
        return Object.IsValid() && Object->TryGetStringField(Key, Value);
    }

    bool IntValue(const TSharedPtr<FJsonObject>& Object, const TCHAR* Key, int32& Value)
    {
        double Number = 0.0;
        if (!Object.IsValid() || !Object->TryGetNumberField(Key, Number)
            || !FMath::IsFinite(Number) || Number < 0
            || Number > static_cast<double>(MAX_int32)
            || FMath::FloorToDouble(Number) != Number)
        {
            return false;
        }
        Value = static_cast<int32>(Number);
        return true;
    }

    bool ReadJsonObject(const TSharedPtr<FJsonObject>& Parent, const TCHAR* Key,
                        TSharedPtr<FJsonObject>& Child)
    {
        if (!Parent.IsValid() || !Parent->HasTypedField<EJson::Object>(Key))
        {
            return false;
        }
        Child = Parent->GetObjectField(Key);
        return Child.IsValid();
    }

    bool ReadRankings(const TSharedPtr<FJsonObject>& Leaderboard, const TCHAR* Key,
                      TArray<FCloudgardenHUDRankRow>& Rows)
    {
        if (!Leaderboard.IsValid() || !Leaderboard->HasTypedField<EJson::Array>(Key))
        {
            return false;
        }
        const TArray<TSharedPtr<FJsonValue>>& Items = Leaderboard->GetArrayField(Key);
        if (Items.Num() > 5) return false;  // Never create more than Top 5.
        for (const TSharedPtr<FJsonValue>& Item : Items)
        {
            if (!Item.IsValid() || Item->Type != EJson::Object) return false;
            const TSharedPtr<FJsonObject> Obj = Item->AsObject();
            FCloudgardenHUDRankRow Row;
            if (!TextValue(Obj, TEXT("plantId"), Row.PlantId)
                || !TextValue(Obj, TEXT("plantName"), Row.PlantName)
                || !TextValue(Obj, TEXT("status"), Row.PlantStatus)
                || !TextValue(Obj, TEXT("netPnlUsdt"), Row.NetPnlUsdt)
                || !TextValue(Obj, TEXT("returnPct"), Row.ReturnPct)
                || !IntValue(Obj, TEXT("closedTrades"), Row.ClosedTrades)
                || Row.PlantId.IsEmpty()) return false;
            Rows.Add(MoveTemp(Row));
        }
        return true;
    }

    bool ReadTiles(const TSharedPtr<FJsonObject>& Root,
                   TArray<FCloudgardenHUDTileRow>& Rows)
    {
        if (!Root->HasTypedField<EJson::Array>(TEXT("tileCards"))) return false;
        const TArray<TSharedPtr<FJsonValue>>& Items = Root->GetArrayField(TEXT("tileCards"));
        if (Items.Num() > 1000) return false;
        for (const TSharedPtr<FJsonValue>& Item : Items)
        {
            if (!Item.IsValid() || Item->Type != EJson::Object) return false;
            const TSharedPtr<FJsonObject> Obj = Item->AsObject();
            FCloudgardenHUDTileRow Row;
            if (!TextValue(Obj, TEXT("tileId"), Row.TileId)
                || !TextValue(Obj, TEXT("label"), Row.Label)
                || !TextValue(Obj, TEXT("capitalMode"), Row.CapitalMode)
                || !TextValue(Obj, TEXT("paperBudgetRequestedUsdt"), Row.PaperBudgetRequestedUsdt)
                || !IntValue(Obj, TEXT("visiblePlantInstances"), Row.VisiblePlantInstances)
                || !Obj->TryGetBoolField(TEXT("evolutionEnabled"), Row.bEvolutionEnabled)
                || !Obj->TryGetBoolField(TEXT("paperBudgetIsFunded"), Row.bPaperBudgetIsFunded)
                || !Obj->TryGetBoolField(TEXT("liveEnabled"), Row.bLiveEnabled)
                || Row.TileId.IsEmpty()
                || Row.bLiveEnabled || Row.bPaperBudgetIsFunded) return false;
            Rows.Add(MoveTemp(Row));
        }
        return true;
    }

    bool ReadMetric(const TSharedPtr<FJsonObject>& Metrics, const TCHAR* Key,
                    FText& Result)
    {
        TSharedPtr<FJsonObject> Obj;
        if (!ReadJsonObject(Metrics, Key, Obj)) return false;
        FString Availability, Value, Unit, Source;
        if (!TextValue(Obj, TEXT("availability"), Availability)
            || !TextValue(Obj, TEXT("unit"), Unit)
            || !TextValue(Obj, TEXT("source"), Source)) return false;
        if (Availability == TEXT("unavailable"))
        {
            Result = FText::FromString(TEXT("—"));
            return Obj->HasTypedField<EJson::Null>(TEXT("value"));
        }
        // This prepared starter intentionally accepts MOCK ONLY. The local
        // agent must add a separately authenticated real-ledger adapter.
        if (Availability != TEXT("mock_only")
            || Source != TEXT("synthetic_only")
            || !TextValue(Obj, TEXT("value"), Value)) return false;
        Result = FText::FromString(Value + TEXT(" ") + Unit);
        return true;
    }
}

bool UCloudgardenHUDWidget::ApplySnapshotJson(const FString& SnapshotJson)
{
    if (SnapshotJson.Len() > 1'000'000) return false;

    TSharedPtr<FJsonObject> Root;
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(SnapshotJson);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid()) return false;

    FString Kind, Authority, Banner, FinancialStatus;
    int32 SchemaVersion = 0;
    if (!IntValue(Root, TEXT("schemaVersion"), SchemaVersion) || SchemaVersion != 1
        || !TextValue(Root, TEXT("kind"), Kind)
        || Kind != TEXT("cloudgarden_ue5_hud_snapshot")
        || !TextValue(Root, TEXT("authority"), Authority)
        || Authority != TEXT("synthetic_hud_preview_only")
        || !TextValue(Root, TEXT("sourceBanner"), Banner)
        || !Banner.Contains(TEXT("DEMO"))
        || !TextValue(Root, TEXT("financialSource"), FinancialStatus)
        || (FinancialStatus != TEXT("unavailable_no_ledger")
            && FinancialStatus != TEXT("synthetic_only"))) return false;

    TSharedPtr<FJsonObject> Metrics, Eco, Budget, Rankings, Focus;
    if (!ReadJsonObject(Root, TEXT("financialMetrics"), Metrics)
        || !ReadJsonObject(Root, TEXT("ecologyMetrics"), Eco)
        || !ReadJsonObject(Root, TEXT("paperBudget"), Budget)
        || !ReadJsonObject(Root, TEXT("leaderboard"), Rankings)
        || !ReadJsonObject(Root, TEXT("focus"), Focus)) return false;

    FText NewNav, NewReturn, NewNet, NewDrawdown, NewWinRate;
    int32 NewCanonical = 0, NewActive = 0, NewInstances = 0, NewEvolving = 0;
    int32 NewReleases = 0, NewLandings = 0, NewChildren = 0;
    FString NewRequested, NewFunded, NewPlantId, NewTileId, LeaderboardScope;
    TArray<FCloudgardenHUDRankRow> NewWinners, NewLosers;
    TArray<FCloudgardenHUDTileRow> NewTiles;

    if (!ReadMetric(Metrics, TEXT("totalBiomassPortfolioUsdt"), NewNav)
        || !ReadMetric(Metrics, TEXT("portfolioReturnPct"), NewReturn)
        || !ReadMetric(Metrics, TEXT("netPnlUsdt"), NewNet)
        || !ReadMetric(Metrics, TEXT("maxDrawdownPct"), NewDrawdown)
        || !ReadMetric(Metrics, TEXT("winRatePct"), NewWinRate)
        || !IntValue(Eco, TEXT("canonicalPlants"), NewCanonical)
        || !IntValue(Eco, TEXT("activePlants"), NewActive)
        || !IntValue(Eco, TEXT("visibleTileRepresentations"), NewInstances)
        || !IntValue(Eco, TEXT("tilesEvolving"), NewEvolving)
        || !IntValue(Eco, TEXT("independentSeedReleases"), NewReleases)
        || !IntValue(Eco, TEXT("referenceSeedLandings"), NewLandings)
        || !IntValue(Eco, TEXT("proposedOffspringCandidates"), NewChildren)
        || !TextValue(Budget, TEXT("requestedTotalUsdt"), NewRequested)
        || !TextValue(Budget, TEXT("fundedTotalUsdt"), NewFunded)
        || NewFunded != TEXT("0.00")
        || !TextValue(Focus, TEXT("selectedPlantId"), NewPlantId)
        || !TextValue(Focus, TEXT("selectedTileId"), NewTileId)
        || !TextValue(Rankings, TEXT("scope"), LeaderboardScope)
        || LeaderboardScope != TEXT("canonical_plant_unique_only")
        || !ReadRankings(Rankings, TEXT("winnerRows"), NewWinners)
        || !ReadRankings(Rankings, TEXT("loserRows"), NewLosers)
        || !ReadTiles(Root, NewTiles)
        || NewCanonical < NewActive) return false;

    // Only publish the read-only parsed view model after every field passed.
    SourceBanner = FText::FromString(Banner);
    FinancialSourceStatus = FinancialStatus;
    PortfolioNavText = NewNav;
    PortfolioReturnText = NewReturn;
    NetPnlText = NewNet;
    MaxDrawdownText = NewDrawdown;
    WinRateText = NewWinRate;
    CanonicalPlantCount = NewCanonical;
    ActivePlantCount = NewActive;
    VisiblePlantInstances = NewInstances;
    TilesEvolving = NewEvolving;
    SeedReleases = NewReleases;
    SeedLandings = NewLandings;
    ProposedOffspring = NewChildren;
    RequestedPaperBudgetUsdt = NewRequested;
    FundedPaperBudgetUsdt = NewFunded;
    SelectedPlantId = NewPlantId;
    SelectedTileId = NewTileId;
    TileRows = MoveTemp(NewTiles);
    WinnerRows = MoveTemp(NewWinners);
    LoserRows = MoveTemp(NewLosers);
    OnHUDDataChanged();
    return true;
}

void UCloudgardenHUDWidget::SetRankingTab(ECloudgardenRankingTab NewTab)
{
    RankingTab = NewTab;
    OnRankingTabChanged();
}

void UCloudgardenHUDWidget::SetExploreCompact(bool bCompact)
{
    bExploreCompact = bCompact;
    OnHUDDataChanged();
}

void UCloudgardenHUDWidget::SetSelectedTile(const FString& TileId)
{
    // UI-only selection; never calls paper, risk or broker services.
    if (!TileRows.ContainsByPredicate([&TileId](const FCloudgardenHUDTileRow& Tile)
        { return Tile.TileId == TileId; })) return;
    SelectedTileId = TileId;
    OnHUDDataChanged();
}

void UCloudgardenHUDWidget::SetSelectedPlant(const FString& PlantId)
{
    // UMG selection only; actor identity resolution belongs to scene subsystem.
    if (PlantId.IsEmpty()) return;
    SelectedPlantId = PlantId;
    OnHUDDataChanged();
}

TArray<FCloudgardenHUDRankRow> UCloudgardenHUDWidget::GetVisibleTopFive() const
{
    return RankingTab == ECloudgardenRankingTab::Winners ? WinnerRows : LoserRows;
}
