#include "SlateCurveChart.h"
#include "Framework/Application/SlateApplication.h"
#include "Rendering/DrawElements.h"
#include "Rendering/SlateRenderer.h"
#include "Styling/CoreStyle.h"

void SCloudgardenCurve::Construct(const FArguments& Args)
{
    Unit = Args._Unit;
    ShowAxes = Args._ShowAxes;
    SetPoints(Args._Points);
    SetVisibility(EVisibility::HitTestInvisible);
}

void SCloudgardenCurve::SetPoints(const TArray<FVector2D>& InPoints)
{
    Points.Reset();
    // The engine retains 300 daily observations. Defend the presentation boundary too.
    for (int32 I = FMath::Max(0, InPoints.Num() - 300); I < InPoints.Num(); ++I)
    {
        if (FMath::IsFinite(InPoints[I].X) && FMath::IsFinite(InPoints[I].Y))
            Points.Add(InPoints[I]);
    }
    Invalidate(EInvalidateWidgetReason::Paint);
}

FVector2D SCloudgardenCurve::ComputeDesiredSize(float) const
{
    return ShowAxes ? FVector2D(420, 144) : FVector2D(240, 58);
}

int32 SCloudgardenCurve::OnPaint(const FPaintArgs&, const FGeometry& Geometry,
    const FSlateRect&, FSlateWindowElementList& Elements, int32 Layer,
    const FWidgetStyle& Style, bool) const
{
    const FLinearColor Mint(.48f, .88f, .74f);
    const FLinearColor Muted(.71f, .81f, .77f);
    const FLinearColor Tint = Style.GetColorAndOpacityTint();
    const FVector2D Size = Geometry.GetLocalSize();
    const FSlateFontInfo Font = FCoreStyle::GetDefaultFontStyle(TEXT("Regular"), 11);
    auto Text = [&](const FString& Value, FVector2D Position, FLinearColor Color)
    {
        FSlateDrawElement::MakeText(Elements, Layer + 3,
            Geometry.ToPaintGeometry(FVector2D(160, 20), FSlateLayoutTransform(Position)),
            Value, Font, ESlateDrawEffect::None, Color * Tint);
    };
    if (Points.IsEmpty())
    {
        Text(TEXT("Noch keine gemessenen Handelstage"), FVector2D(0, Size.Y * .45), Muted);
        return Layer + 3;
    }

    double Low = Points[0].Y, High = Low;
    for (const FVector2D& Point : Points)
    {
        Low = FMath::Min(Low, Point.Y);
        High = FMath::Max(High, Point.Y);
    }
    const double Pad = FMath::Max(.02, (High - Low) * .12);
    Low -= Pad;
    High += Pad;
    const float Left = ShowAxes ? 52.f : 2.f;
    const float Top = 8.f;
    const float Width = FMath::Max(1.f, float(Size.X) - Left - 10.f);
    const float Height = FMath::Max(1.f, float(Size.Y) - Top - (ShowAxes ? 25.f : 5.f));
    const double Span = Points.Last().X - Points[0].X;
    TArray<FVector2D> Curve;
    Curve.Reserve(Points.Num());
    for (const FVector2D& Point : Points)
        Curve.Add(FVector2D(Left + Width * (Span > 0 ? (Point.X - Points[0].X) / Span : .5),
            Top + Height * (High - Point.Y) / (High - Low)));

    FNumberFormattingOptions Numbers;
    Numbers.SetMinimumFractionalDigits(2);
    Numbers.SetMaximumFractionalDigits(2);
    auto Number = [&](double V) { return FText::AsNumber(V, &Numbers).ToString(); };
    if (ShowAxes)
    {
        for (int32 I = 0; I < 3; ++I)
        {
            const float Y = Top + Height * I / 2.f;
            FSlateDrawElement::MakeLines(Elements, Layer,
                Geometry.ToPaintGeometry(), TArray<FVector2D>{FVector2D(Left, Y), FVector2D(Left + Width, Y)},
                ESlateDrawEffect::None, FLinearColor(.5f, .7f, .63f, .18f) * Tint, true, .7f);
            Text(Number(High - (High - Low) * I / 2), FVector2D(0, Y - 6), Muted);
        }
        Text(FString::Printf(TEXT("Tag %.0f"), Points[0].X), FVector2D(Left, Top + Height + 8), Muted);
        if (Points.Num() > 1)
            Text(FString::Printf(TEXT("Tag %.0f"), Points.Last().X),
                FVector2D(FMath::Max(double(Left), Size.X - 72), Top + Height + 8), Muted);
    }

    if (Curve.Num() > 1)
    {
        // Straight segments preserve measured values; no spline invents interim extrema.
        TArray<FSlateVertex> Vertices;
        TArray<SlateIndex> Indices;
        for (int32 I = 0; I < Curve.Num(); ++I)
        {
            const FVector2f Upper(Curve[I]);
            const FVector2f Lower(Curve[I].X, Top + Height);
            Vertices.Add(FSlateVertex::Make<ESlateVertexRounding::Disabled>(
                Geometry.GetAccumulatedRenderTransform(), Upper, FVector2f(.5f, .5f),
                (FLinearColor(Mint.R, Mint.G, Mint.B, .22f) * Tint).ToFColor(true)));
            Vertices.Add(FSlateVertex::Make<ESlateVertexRounding::Disabled>(
                Geometry.GetAccumulatedRenderTransform(), Lower, FVector2f(.5f, .5f),
                FLinearColor(Mint.R, Mint.G, Mint.B, 0).ToFColor(true)));
            if (I > 0)
            {
                const SlateIndex A = (I - 1) * 2, B = I * 2;
                Indices.Append({A, SlateIndex(A + 1), B, B, SlateIndex(A + 1), SlateIndex(B + 1)});
            }
        }
        const FSlateBrush* White = FCoreStyle::Get().GetBrush(TEXT("WhiteBrush"));
        FSlateDrawElement::MakeCustomVerts(Elements, Layer + 1,
            FSlateApplication::Get().GetRenderer()->GetResourceHandle(*White), Vertices, Indices,
            nullptr, 0, 0);
        FSlateDrawElement::MakeLines(Elements, Layer + 2, Geometry.ToPaintGeometry(), Curve,
            ESlateDrawEffect::None, Mint * Tint, true, 1.6f);
    }

    const FVector2D Last = Curve.Last();
    TArray<FVector2D> Dot;
    for (int32 I = 0; I <= 16; ++I)
    {
        const double A = I * 2 * PI / 16;
        Dot.Add(Last + FVector2D(FMath::Cos(A), FMath::Sin(A)) * 2.2);
    }
    FSlateDrawElement::MakeLines(Elements, Layer + 3, Geometry.ToPaintGeometry(), Dot,
        ESlateDrawEffect::None, FLinearColor(.87f, 1.f, .95f) * Tint, true, 2.f);
    if (Points.Num() == 1)
        Text(Number(Points.Last().Y) + TEXT(" ") + Unit,
            FVector2D(Left + 8, FMath::Max(0.f, Top + Height * .18f)), Muted);
    return Layer + 3;
}
