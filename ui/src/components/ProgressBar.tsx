export interface ProgressState {
  phase: string;
  done: number;
  total: number;
  detail?: string;
}

export function ProgressBar({
  progress,
  fallbackLabel = "working…",
}: {
  progress: ProgressState | null;
  fallbackLabel?: string;
}) {
  if (!progress) {
    return (
      <div className="progress-wrap">
        <div className="progress-meta muted">
          <span className="spinner" /> {fallbackLabel}
        </div>
        <div className="progress-track indeterminate">
          <div className="progress-fill" />
        </div>
      </div>
    );
  }
  const { phase, done, total, detail } = progress;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  const label =
    total > 0
      ? `${phase} ${done}/${total}${detail ? ` · ${detail}` : ""}`
      : `${phase}${detail ? ` · ${detail}` : ""}`;

  return (
    <div className="progress-wrap">
      <div className="progress-meta">
        <span className="progress-label">{label}</span>
        {total > 0 && <span className="progress-pct muted">{pct}%</span>}
      </div>
      <div className={`progress-track ${total > 0 ? "" : "indeterminate"}`}>
        <div className="progress-fill" style={total > 0 ? { width: `${pct}%` } : undefined} />
      </div>
    </div>
  );
}
