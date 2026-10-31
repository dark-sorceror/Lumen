"use client";

import { useEffect, useRef } from "react";
import { glideY, inPageId } from "./glide";
import { reducedMotion } from "./motion";

/** In-page links glide to their section and stop just under the sticky nav
 * (`navSelector`); the logo's "#top" goes to the very top. The page scrolls
 * inside its own root (the nearest `[data-scroller]`), not the document.
 * A link with `data-show="app" | "plugin"` first asks `onShow` to switch the
 * hero's view. Focus lands on the section's heading, without scrolling.
 * Modified clicks and links to ids that aren't on the page are left alone. */
export function useGlideLinks(navSelector: string, onShow: (view: "app" | "plugin") => void) {
  const show = useRef(onShow);
  useEffect(() => {
    show.current = onShow;
  });

  useEffect(() => {
    function onClick(e: MouseEvent) {
      const a = (e.target as Element | null)?.closest?.('a[href^="#"]');
      if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const id = inPageId(a.getAttribute("href"));
      if (id === null) return;
      const target = id === "top" ? null : document.getElementById(id);
      if (id !== "top" && !target) return;
      e.preventDefault();
      const view = (a as HTMLElement).dataset.show;
      if (view === "app" || view === "plugin") show.current(view);

      const nav = document.querySelector(navSelector);
      const scroller = nav?.closest("[data-scroller]") ?? null;
      const navHeight = nav?.getBoundingClientRect().height ?? 0;
      const behavior: ScrollBehavior = reducedMotion() ? "auto" : "smooth";
      if (scroller) {
        const top = target ? target.getBoundingClientRect().top - scroller.getBoundingClientRect().top : null;
        scroller.scrollTo({ top: glideY(top, scroller.scrollTop, navHeight), behavior });
      } else {
        window.scrollTo({ top: glideY(target ? target.getBoundingClientRect().top : null, window.scrollY, navHeight), behavior });
      }
      const land = (target ? target.querySelector<HTMLElement>("h2, h1") || target : document.getElementById("hero-title")) as HTMLElement | null;
      if (land) {
        land.setAttribute("tabindex", "-1");
        land.focus({ preventScroll: true });
      }
    }
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [navSelector]);
}
