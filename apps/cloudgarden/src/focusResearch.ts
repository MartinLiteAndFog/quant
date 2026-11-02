import {createHistoricalSession,type HistoricalSession,type SolArchive} from './historicalGarden.ts';
import {assertSimulationInvariants} from './simulation.ts';
/** Separate new research world: one funded parent, no births until explicitly enabled. */
export function createFocusedHistoricalSession(archive:SolArchive):HistoricalSession {
 const session=createHistoricalSession(archive);
 const plants=session.world.plants.map((plant,index)=>index===0?plant:{...plant,status:'archived' as const,allocationUsdt:0});
 const released=session.world.plants.slice(1).reduce((sum,p)=>sum+p.allocationUsdt,0);
 const world={...session.world,evolutionPaused:true,plants,capital:{...session.world.capital,deployedCapitalUsdt:session.world.capital.deployedCapitalUsdt-released,freeCapitalUsdt:session.world.capital.freeCapitalUsdt+released}};
 assertSimulationInvariants(world);
 return {...session,world};
}
