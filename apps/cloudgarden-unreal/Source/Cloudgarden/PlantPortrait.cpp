#include "PlantPortrait.h"
#include "Cloudgarden.h"
#include "Engine/SceneCapture2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "Rendering/DrawElements.h"

void SCloudgardenPlantPortrait::Construct(const FArguments& Args)
{
    Owner = Args._OwnerHUD;
    Brush.DrawAs = ESlateBrushDrawType::Image;
    Brush.ImageSize = FVector2D(160, 192);
    SetVisibility(EVisibility::HitTestInvisible);
}

SCloudgardenPlantPortrait::~SCloudgardenPlantPortrait()
{
    if (Capture.IsValid()) Capture->Destroy();
}

FVector2D SCloudgardenPlantPortrait::ComputeDesiredSize(float) const
{
    return FVector2D(150, 180);
}

void SCloudgardenPlantPortrait::Tick(const FGeometry& G, double Time, float Delta)
{
    SLeafWidget::Tick(G, Time, Delta);
    ACloudgardenHUD* HUD = Owner.Get();
    if (!HUD || !HUD->GetWorld()) return;
    auto PC = Cast<ACloudgardenController>(HUD->GetOwningPlayerController());
    ACloudgardenPlant* Plant = PC ? PC->Selected : nullptr;
    if (!IsValid(Plant))
    {
        HasPortrait = false;
        LastPlant.Reset();
        LastData.Reset();
        return;
    }
    if (LastPlant.Get() == Plant && LastData.Pin() == Plant->Data) return;
    LastPlant = Plant;
    LastData = Plant->Data;

    if (!Capture.IsValid())
    {
        FActorSpawnParameters Params;
        Params.ObjectFlags |= RF_Transient;
        Capture = HUD->GetWorld()->SpawnActor<ASceneCapture2D>(Params);
        if (!Capture.IsValid()) return;
        auto Component = Capture->GetCaptureComponent2D();
        Component->bCaptureEveryFrame = false;
        Component->bCaptureOnMovement = false;
        Component->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
        Component->CaptureSource = ESceneCaptureSource::SCS_SceneColorHDR;
        Component->ProjectionType = ECameraProjectionMode::Orthographic;
        Component->ShowFlags.SetAtmosphere(false);
        Component->ShowFlags.SetFog(false);
        Component->ShowFlags.SetVolumetricFog(false);
        Component->ShowFlags.SetMotionBlur(false);
        Target.Reset(NewObject<UTextureRenderTarget2D>(GetTransientPackage()));
        Target->RenderTargetFormat = RTF_RGBA16f;
        // SceneColor HDR stores inverse opacity. Slate inverts that channel when drawing.
        Target->ClearColor = FLinearColor(0, 0, 0, 1);
        Target->InitAutoFormat(320, 384);
        Component->TextureTarget = Target.Get();
        Brush.SetResourceObject(Target.Get());
    }

    auto Component = Capture->GetCaptureComponent2D();
    Component->ShowOnlyActors.Reset();
    FBox Bounds(ForceInit);
    for (TActorIterator<AActor> It(HUD->GetWorld()); It; ++It)
    {
        if (!It->IsHidden() && (*It == Plant || It->GetOwner() == Plant))
        {
            Component->ShowOnlyActors.Add(*It);
            Bounds += It->GetComponentsBoundingBox();
        }
    }
    if (!Bounds.IsValid) return;
    const FVector Centre = Bounds.GetCenter();
    const FVector Extent = Bounds.GetSize();
    Component->OrthoWidth = FMath::Max(90., FMath::Max(FMath::Max(Extent.X, Extent.Y), Extent.Z * 320. / 384.) * 1.18);
    Capture->SetActorLocation(Centre + FVector(450, -450, 150));
    Capture->SetActorRotation((Centre - Capture->GetActorLocation()).Rotation());
    Component->CaptureScene();
    HasPortrait = true;
    Invalidate(EInvalidateWidgetReason::Paint);
}

int32 SCloudgardenPlantPortrait::OnPaint(const FPaintArgs&, const FGeometry& G,
    const FSlateRect&, FSlateWindowElementList& Elements, int32 Layer,
    const FWidgetStyle& Style, bool) const
{
    if (HasPortrait && Target.IsValid())
    {
        const FVector2D Size = G.GetLocalSize();
        const double Scale = FMath::Min(Size.X / 160., Size.Y / 192.);
        const FVector2D ImageSize(160 * Scale, 192 * Scale);
        FSlateDrawElement::MakeBox(Elements, Layer,
            G.ToPaintGeometry(ImageSize, FSlateLayoutTransform((Size - ImageSize) * .5)),
            &Brush, ESlateDrawEffect::InvertAlpha, Style.GetColorAndOpacityTint());
    }
    return Layer;
}
