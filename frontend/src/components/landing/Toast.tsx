"use client";

import { createContext, type ReactNode, useCallback, useContext, useEffect, useRef, useState } from "react";
import kit from "@/app/(landing)/page.module.css";

const SHOWN_MS = 2800;

const ToastContext = createContext<(message: string) => void>(() => {});

/** Says a short message at the foot of the page for a moment. The region is
 * always there, so a screen reader announces each new message. */
export function useToast(): (message: string) => void {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [message, setMessage] = useState<{ text: string; n: number } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);

  const say = useCallback((text: string) => {
    setMessage((m) => ({ text, n: (m?.n ?? 0) + 1 }));
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setMessage(null), SHOWN_MS);
  }, []);

  return (
    <ToastContext.Provider value={say}>
      {children}
      <div className={kit.toastHost} role="status">
        {message && (
          <div className={kit.toast} key={message.n}>
            {message.text}
          </div>
        )}
      </div>
    </ToastContext.Provider>
  );
}
