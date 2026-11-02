#include "CloudgardenPlantFocusWidget.h"

#include "Components/Image.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
    bool ValidInt(const TSharedPtr<FJsonObject>& Source, const TCHAR* Key, int32& Out)
    {
        double Value = 0.0;
        if (!Source.IsValid() || !Source->TryGetNumberField(Key, Value)
            || !FMath::IsFinite(Value) || Value < 0
            || Value > MAX_int32 || FMath::Floor(Value) != Value) return false;
        Out = static_cast<int32>(Value);
        return true;
    }

    bool MoneyOrUnavailable(const TSharedPtr<FJsonObject>& Source,
                            const TCHAR* Key, FString& Out)
    {
        if (!Source.IsValid() || !Source->HasField(Key)) return false;
        const TSharedPtr<FJsonValue> Value = Source->TryGetField(Key);
        if (!Value.IsValid()) return false;
        if (Value->Type == EJson::Null)
        {
            Out = TEXT("—");
            return true;
        }
        if (Value->Type != EJson::String) return false;
        Out = Value->AsString();
        return !Out.IsEmpty() && Out.Len() <= 64;
    }
}

bool UCloudgardenPlantFocusWidget::ImportSyntheticHUDJson(const FString& HudJson)
{
    if (HudJson.IsEmpty() || HudJson.Len() > 1'000'000) return false;

    TSharedPtr<FJsonObject> Root;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(HudJson), Root)
        || !Root.IsValid()) return false;
    FString Kind, Authority, Banner;
    int32 Schema = 0;
    if (!ValidInt(Root, TEXT("schemaVersion"), Schema) || Schema != 1
        || !Root->TryGetStringField(TEXT("kind"), Kind)
        || Kind != TEXT("cloudgarden_ue5_hud_snapshot")
        || !Root->TryGetStringField(TEXT("authority"), Authority)
        || Authority != TEXT("synthetic_hud_preview_only")
        || !Root->TryGetStringField(TEXT("sourceBanner"), Banner)
        || !Banner.Contains(TEXT("DEMO"))
        || !Root->HasTypedField<EJson::Object>(TEXT("plantInspectors"))) return false;

    const TSharedPtr<FJsonObject> Index = Root->GetObjectField(TEXT("plantInspectors"));
    if (!Index.IsValid() || Index->Values.Num() > 10000) return false;
    TMap<FString, FCloudgardenPlantFocusData> Next;
    for (const auto& Entry : Index->Values)
    {
        if (!Entry.Value.IsValid() || Entry.Value->Type != EJson::Object) return false;
        const TSharedPtr<FJsonObject> Source = Entry.Value->AsObject();
        if (!Source.IsValid() || !Source->HasTypedField<EJson::Object>(TEXT("metrics"))
            || !Source->HasTypedField<EJson::Array>(TEXT("tileInstanceIds"))
            || !Source->HasTypedField<EJson::Array>(TEXT("equitySeries"))) return false;
        FCloudgardenPlantFocusData Row;
        if (!Source->TryGetStringField(TEXT("plantId"), Row.PlantId)
            || Row.PlantId != Entry.Key || Row.PlantId.IsEmpty()
            || !Source->TryGetStringField(TEXT("name"), Row.PlantName)
            || !Source->TryGetStringField(TEXT("status"), Row.LifeStatus)
            || !Source->TryGetStringField(TEXT("financialSource"), Row.FinancialSource)
            || (Row.FinancialSource != TEXT("unavailable_no_plant_ledger")
                && Row.FinancialSource != TEXT("fictional_per_plant_equity_only"))
            || !ValidInt(Source, TEXT("ageDays"), Row.AgeDays)
            || !ValidInt(Source, TEXT("generation"), Row.Generation)) return false;

        const TSharedPtr<FJsonObject> Metrics = Source->GetObjectField(TEXT("metrics"));
        if (!MoneyOrUnavailable(Metrics, TEXT("individualEquityUsdt"), Row.IndividualEquityUsdt)
            || !MoneyOrUnavailable(Metrics, TEXT("netPnlUsdt"), Row.NetPnlUsdt)
            || !MoneyOrUnavailable(Metrics, TEXT("highWaterMarkUsdt"), Row.HighWaterMarkUsdt)
            || !MoneyOrUnavailable(Metrics, TEXT("maxDrawdownPct"), Row.MaxDrawdownPct)
            || !MoneyOrUnavailable(Metrics, TEXT("winRatePct"), Row.WinRatePct)) return false;

        Row.MatchingTileCopies = Source->GetArrayField(TEXT("tileInstanceIds")).Num();
        if (Row.MatchingTileCopies > 10000) return false;
        const TArray<TSharedPtr<FJsonValue>>& Series = Source->GetArrayField(TEXT("equitySeries"));
        if (Series.Num() > 10000) return false;
        for (const TSharedPtr<FJsonValue>& Point : Series)
        {
            if (!Point.IsValid() || Point->Type != EJson::Object) return false;
            const TSharedPtr<FJsonObject> Obj = Point->AsObject();
            FCloudgardenPlantEquityPoint Record;
            if (!ValidInt(Obj, TEXT("dayIndex"), Record.DayIndex)
                || !Obj->TryGetStringField(TEXT("equityUsdt"), Record.EquityUsdt)) return false;
            Row.EquityPoints.Add(MoveTemp(Record));
        }
        if ((Row.FinancialSource == TEXT("unavailable_no_plant_ledger") && !Row.EquityPoints.IsEmpty())
            || (Row.FinancialSource == TEXT("fictional_per_plant_equity_only") && Row.EquityPoints.IsEmpty()))
            return false;
        Next.Add(Entry.Key, MoveTemp(Row));
    }
    // Publish only after all rows were read. Caller still needs the separately
    // verified bootstrap digest before passing this JSON to the widget.
    SourcePlantRows = MoveTemp(Next);
    if (!SelectedSourcePlantId.IsEmpty() && SourcePlantRows.Contains(SelectedSourcePlantId))
    {
        SelectedPlant = SourcePlantRows.FindChecked(SelectedSourcePlantId);
        OnFocusChanged();
    }
    else
    {
        bDrawerOpen = false;
    }
    return true;
}

bool UCloudgardenPlantFocusWidget::FocusPlantById(const FString& CanonicalPlantId)
{
    const FCloudgardenPlantFocusData* Focus = SourcePlantRows.Find(CanonicalPlantId);
    if (!Focus) return false;
    SelectedSourcePlantId = CanonicalPlantId;
    SelectedPlant = *Focus;
    bDrawerOpen = true;
    SetVisibility(ESlateVisibility::Visible);
    OnFocusChanged();
    // Cosmetic ease only; no teleport, reset, trade or additional plant.
    RequestSoftWorldCameraFocus(CanonicalPlantId, 0.65f, 0.12f);
    return true;
}

void UCloudgardenPlantFocusWidget::CloseFocusDrawer()
{
    bDrawerOpen = false;
    SetVisibility(ESlateVisibility::Collapsed);
    OnFocusChanged();
    RequestRestoreWorldCamera();
}

void UCloudgardenPlantFocusWidget::SetPlantPreviewRenderTarget(UTextureRenderTarget2D* RenderTarget)
{
    // Render target comes from preview mesh in a separate cosmetic capture stage.
    // The preview scene must have NO physics, collision, seed emitter or ledger.
    if (PlantPreviewImage && RenderTarget)
    {
        PlantPreviewImage->SetBrushResourceObject(RenderTarget);
    }
}
