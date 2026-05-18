import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Avoid refetch storms when the user toggles back to the tab — let the
      // existing refetchInterval drive cadence instead.
      refetchOnWindowFocus: false,
      retry: 1,
      // Match the backend chart cache TTL so we don't issue an immediate fresh
      // request on every component remount.
      staleTime: 4000,
      // Keep gc generous so range/equityMode toggles don't refetch instantly.
      gcTime: 5 * 60 * 1000,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
