import { useEffect, useState } from "react";

const ONBOARD_KEY = "membukkit-onboarded";
const ASK_TIP_KEY = "membukkit-ask-tip-dismissed";

export function hasCompletedOnboarding(): boolean {
  return localStorage.getItem(ONBOARD_KEY) === "1";
}

export function resetOnboarding(): void {
  localStorage.removeItem(ONBOARD_KEY);
  localStorage.removeItem(ASK_TIP_KEY);
}

/** First-visit welcome — one screen, then gone. */
export function OnboardingWelcome({
  onTryDemo,
  onDismiss,
}: {
  onTryDemo?: () => void;
  onDismiss: () => void;
}) {
  const finish = () => {
    localStorage.setItem(ONBOARD_KEY, "1");
    onDismiss();
  };

  return (
    <div className="modal-backdrop onboard-backdrop">
      <div className="modal onboard-modal" onClick={(e) => e.stopPropagation()}>
        <h3>
          Memory with <span className="accent">receipts</span>
        </h3>
        <ol className="onboard-steps">
          <li>
            <strong>Demos</strong> — load a scene and ask the same question at two dates.
          </li>
          <li>
            <strong>As of</strong> — answers use only facts known by that date.
          </li>
          <li>
            <strong>Receipts</strong> — click evidence on the right to see the source.
          </li>
        </ol>
        <p className="muted onboard-note">
          Everything stays on this machine under <code>~/.membukkit</code>. Paste an API key via{" "}
          <strong>keys</strong> in the sidebar if prompted.
        </p>
        <div className="confirm-actions">
          <button type="button" className="ghost" onClick={finish}>
            explore on my own
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => {
              finish();
              onTryDemo?.();
            }}
          >
            show demos
          </button>
        </div>
      </div>
    </div>
  );
}

/** One-time tip under the Ask form after the user has a store. */
export function AskTip() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    setShow(localStorage.getItem(ASK_TIP_KEY) !== "1");
  }, []);

  if (!show) return null;

  return (
    <div className="ask-tip" role="note">
      <p>
        <strong>As of</strong> answers as if that day were today.{" "}
        <strong>Receipts</strong> on the right list cost and the memories used — click one for
        source.
      </p>
      <button
        type="button"
        className="ghost"
        onClick={() => {
          localStorage.setItem(ASK_TIP_KEY, "1");
          setShow(false);
        }}
      >
        got it
      </button>
    </div>
  );
}
