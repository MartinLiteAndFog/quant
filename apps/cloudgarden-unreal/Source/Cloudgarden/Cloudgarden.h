#pragma once
#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/HUD.h"
#include "Dom/JsonObject.h"
#include "Cloudgarden.generated.h"

UCLASS()
class ACloudgardenPlant : public AActor {
 GENERATED_BODY()
public:
 ACloudgardenPlant();
 UPROPERTY() UStaticMeshComponent* Mesh;
 FString PlantId, PlantName, GardenSymbol;
 FVector ExactAnchor;
 FVector CrownOffset;
 TSharedPtr<FJsonObject> Data;
};
UCLASS()
class ACloudgardenController : public APlayerController {
 GENERATED_BODY()
public:
 virtual void BeginPlay() override;
 virtual void PlayerTick(float Delta) override;
 UPROPERTY() ACloudgardenPlant* Selected;
 bool WasDown=false, WasSpace=false, WasRight=false;
 int InspectorScroll=0;
 bool ShowDetails=false, ShowLibrary=false;
 int CatalogIndex=0;
};
UCLASS()
class ACloudgardenHUD : public AHUD {
 GENERATED_BODY()
public:
 virtual void DrawHUD() override;
};
UCLASS()
class ACloudgardenMode : public AGameModeBase {
 GENERATED_BODY()
public:
 ACloudgardenMode();
 virtual void BeginPlay() override;
 virtual void Tick(float Delta) override;
 void LoadSnapshot();
 void CollectGene(int Index);
 int Plane=0;
 bool MotionEnabled=true;
 TSharedPtr<FJsonObject> Catalog;
 struct Flight { AActor* Actor; FVector From,To; float Age; };
 TArray<Flight> Flights;
 void SendCommand(const FString& Action);
 bool Playing=false;
 TSharedPtr<FJsonObject> Snapshot;
 FString Status=TEXT("Lokales Datenpaket wird geladen …");
 TArray<AActor*> Generated;
 FTimerHandle RefreshTimer;
 FString LastPayload;
};
