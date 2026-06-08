import { Suspense } from "react";
import { CompareInner } from "./compare-inner";

export default function ComparePage() {
  return (
    <Suspense
      fallback={
        <p className="text-sm text-[color:var(--muted)]">Loading…</p>
      }
    >
      <CompareInner />
    </Suspense>
  );
}
