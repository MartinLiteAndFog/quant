import { HISTORICAL_SAVE_KEY } from './historicalGarden';
const DB_NAME='cloudgarden-worlds', STORE='snapshots';
/** IndexedDB keeps long-lived populations outside the small localStorage text quota. */
function database():Promise<IDBDatabase>{
  return new Promise((resolve,reject)=>{
    const request=indexedDB.open(DB_NAME,1);
    request.onupgradeneeded=()=>{if(!request.result.objectStoreNames.contains(STORE))request.result.createObjectStore(STORE);};
    request.onsuccess=()=>resolve(request.result);
    request.onerror=()=>reject(request.error??new Error('Lokaler Weltspeicher nicht verfügbar.'));
    request.onblocked=()=>reject(new Error('Ein anderes Fenster blockiert den Weltspeicher.'));
  });
}
export async function loadHistoricalSnapshot(key = HISTORICAL_SAVE_KEY):Promise<unknown|null>{
  const db=await database();
  try{
    const value=await new Promise<unknown>((resolve,reject)=>{
      const request=db.transaction(STORE,'readonly').objectStore(STORE).get(key);
      request.onsuccess=()=>resolve(request.result??null);request.onerror=()=>reject(request.error);
    });
    if(value!==null)return value;
    const legacy=localStorage.getItem(key);
    return legacy?JSON.parse(legacy):null;
  }finally{db.close();}
}
async function writeSnapshot(snapshot:unknown,key:string):Promise<void>{
  const db=await database();
  try{await new Promise<void>((resolve,reject)=>{
    const transaction=db.transaction(STORE,'readwrite');
    transaction.objectStore(STORE).put(snapshot,key);
    transaction.oncomplete=()=>resolve();transaction.onerror=()=>reject(transaction.error);
    transaction.onabort=()=>reject(transaction.error??new Error('Speichern abgebrochen.'));
  });}finally{db.close();}
}

let saveQueue:Promise<void>=Promise.resolve();
export function saveHistoricalSnapshot(snapshot:unknown,key=HISTORICAL_SAVE_KEY):Promise<void>{
  saveQueue=saveQueue.catch(()=>{}).then(()=>writeSnapshot(snapshot,key));
  return saveQueue;
}
