#include "HUDScrim.h"
#include "Rendering/DrawElements.h"
#include "Framework/Application/SlateApplication.h"
#include "Rendering/SlateRenderer.h"
#include "Styling/CoreStyle.h"

void SCloudgardenHUDScrim::Construct(const FArguments&)
{
    SetVisibility(EVisibility::HitTestInvisible);
}

FVector2D SCloudgardenHUDScrim::ComputeDesiredSize(float) const
{
    return FVector2D::ZeroVector;
}

int32 SCloudgardenHUDScrim::OnPaint(const FPaintArgs&, const FGeometry& Geometry,
    const FSlateRect&, FSlateWindowElementList& Elements, int32 Layer,
    const FWidgetStyle& Style, bool) const
{
    const FVector2D Size = Geometry.GetLocalSize();
    TArray<FSlateVertex> Vertices;
    TArray<SlateIndex> Indices;
    constexpr int32 Segments = 64;
    constexpr int32 Rings = 8;
    auto Ellipse = [&](FVector2D Centre, FVector2D Radius, float Opacity)
    {
        const int32 First = Vertices.Num();
        // Shared centre fan; concentric rings fade gently to transparent curved edges.
        auto Vertex = [&](FVector2D Position, float Alpha)
        {
            Vertices.Add(FSlateVertex::Make<ESlateVertexRounding::Disabled>(
                Geometry.GetAccumulatedRenderTransform(), FVector2f(Position), FVector2f(.5f, .5f),
                (FLinearColor(.045f, .065f, .075f, Alpha) * Style.GetColorAndOpacityTint()).ToFColor(true)));
        };
        Vertex(Centre, Opacity);
        for (int32 Ring = 1; Ring <= Rings; ++Ring)
        {
            const float R = float(Ring) / Rings;
            const float Alpha = Opacity * FMath::Square(1.f - R);
            for (int32 Segment = 0; Segment < Segments; ++Segment)
            {
                const double A = Segment * 2 * PI / Segments;
                Vertex(Centre + FVector2D(FMath::Cos(A) * Radius.X, FMath::Sin(A) * Radius.Y) * R, Alpha);
                const SlateIndex Current = First + 1 + (Ring - 1) * Segments + Segment;
                const SlateIndex Next = First + 1 + (Ring - 1) * Segments + (Segment + 1) % Segments;
                if (Ring == 1) Indices.Append({SlateIndex(First), Current, Next});
                else
                {
                    const SlateIndex Previous = Current - Segments, PreviousNext = Next - Segments;
                    Indices.Append({Previous, Current, Next, Previous, Next, PreviousNext});
                }
            }
        }
    };
    Ellipse(FVector2D(Size.X * .10, Size.Y * .28), FVector2D(Size.X * .31, Size.Y * .50), .16f);
    Ellipse(FVector2D(Size.X * .90, Size.Y * .28), FVector2D(Size.X * .31, Size.Y * .50), .16f);
    Ellipse(FVector2D(Size.X * .50, Size.Y * .98), FVector2D(Size.X * .74, Size.Y * .17), .18f);
    const FSlateBrush* White = FCoreStyle::Get().GetBrush(TEXT("WhiteBrush"));
    FSlateDrawElement::MakeCustomVerts(Elements, Layer,
        FSlateApplication::Get().GetRenderer()->GetResourceHandle(*White), Vertices, Indices, nullptr, 0, 0);
    return Layer;
}
