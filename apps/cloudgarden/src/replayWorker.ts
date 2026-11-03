import { handleSyntheticReplayWorkerMessage, type SyntheticReplayCycleInput } from "./replayCycle.ts";

type ReplayWorkerRequest = Readonly<{ id: string; input: SyntheticReplayCycleInput }>;
type ReplayWorkerScope = {
  onmessage: ((event: MessageEvent<ReplayWorkerRequest>) => void) | null;
  postMessage: (message: unknown) => void;
};

const replayWorker = globalThis as ReplayWorkerScope;
replayWorker.onmessage = (event: MessageEvent<ReplayWorkerRequest>) => {
  void handleSyntheticReplayWorkerMessage(event.data).then((response) => {
    replayWorker.postMessage(response);
  });
};
