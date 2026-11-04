"use client";

import { type RefObject, useEffect, useRef } from "react";
import { reducedMotion } from "./motion";
import { landAt, pickStep } from "./stepLine";

/** A phone, where the story's figure sits over its steps rather than beside
 * them: StorySection.module.css's own width. */
export const NARROW = "(max-width: 900px)";
/** Too short a screen to read beside or under the figure (a phone on its
 * side, a short window): the story stays as it was, with nothing pinned,
 * and a step is chosen by its button, not the scroll. The stylesheet's own
 * query, word for word; it pins the figure at its exact negation,
 * `(height > 480px)`, so no height falls between. */
export const UNPINNED = "(max-height: 480px)";

/** The story's parts the scroll is read from. */
export type StoryParts = {
  story: RefObject<HTMLElement | null>;
  steps: RefObject<(HTMLElement | null)[]>;
  stage: RefObject<HTMLElement | null>;
};

/** Which of the story's steps the reader is on, from 0, told to `onStep`
 * as it changes: the last whose words have reached a line across the screen
 * (pickStep, in stepLine.ts). Beside the figure a step's middle counts;
 * where the figure pins over the steps, a step's top, a fixed distance below
 * the figure's real bottom. `first` is true for the first choice after the
 * page opens or the figure pins again, which the figure takes without
 * travelling there. It listens to the scroll only while the story is near,
 * and not at all where the figure doesn't pin. It reads on the scroll event
 * itself, which a browser sends once a frame, so it keeps no frame of its
 * own. The figure's height goes on the story as --stage-h, so a step that
 * focus scrolls to lands under it. */
export function useActiveStep(
  { story: storyRef, steps: stepRefs, stage: stageRef }: StoryParts,
  onStep: (step: number, first: boolean) => void,
): void {
  const tell = useRef(onStep);
  useEffect(() => {
    tell.current = onStep;
  });

  useEffect(() => {
    const story = storyRef.current;
    if (!story) return;
    const narrow = window.matchMedia(NARROW);
    const unpinned = window.matchMedia(UNPINNED);
    // The step chosen last, for pickStep's hysteresis; null until the first
    // choice, and again while the figure doesn't pin.
    let chosen: number | null = null;
    const pick = () => {
      if (unpinned.matches) {
        chosen = null;
        return;
      }
      const stage = stageRef.current;
      const boxes = stepRefs.current.map((el) => el?.getBoundingClientRect() ?? null);
      const next = pickStep({
        // A step not drawn yet is never reached.
        tops: boxes.map((box) => box?.top ?? Infinity),
        heights: boxes.map((box) => box?.height ?? 0),
        vh: window.innerHeight,
        pinnedBottom: narrow.matches && stage ? stage.getBoundingClientRect().bottom : null,
        current: chosen,
      });
      if (next === chosen) return;
      const first = chosen === null;
      chosen = next;
      tell.current(next, first);
    };
    // Captured on the document, so it hears the landing's own scroller
    // (page.module.css), which the document's scroll never moves.
    let listening = false;
    const listen = (on: boolean) => {
      if (on === listening) return;
      listening = on;
      if (on) {
        document.addEventListener("scroll", pick, { passive: true, capture: true });
        window.addEventListener("resize", pick);
      } else {
        document.removeEventListener("scroll", pick, { capture: true });
        window.removeEventListener("resize", pick);
      }
    };
    pick();
    if (typeof IntersectionObserver === "undefined") {
      listen(true);
      return () => listen(false);
    }
    const observer = new IntersectionObserver(
      (entries) => {
        const near = entries[entries.length - 1].isIntersecting;
        listen(near);
        if (near) pick();
      },
      { rootMargin: "25% 0px 25% 0px" },
    );
    observer.observe(story);
    return () => {
      observer.disconnect();
      listen(false);
    };
  }, [storyRef, stepRefs, stageRef]);

  useEffect(() => {
    const stage = stageRef.current;
    const story = storyRef.current;
    if (!stage || !story || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => story.style.setProperty("--stage-h", `${stage.offsetHeight}px`));
    observer.observe(stage);
    return () => observer.disconnect();
  }, [storyRef, stageRef]);
}

/** Scrolls the landing so step `i` is the one read (landAt, in stepLine.ts),
 * and says whether it did: where the figure doesn't pin, the step is the
 * caller's to show. The landing scrolls inside its own root, as its links do
 * (useGlideLinks.ts): smoothly, or at once with reduced motion. The figure
 * then follows the scroll, step by step; focus stays where it is. */
export function glideToStep(i: number, { story: storyRef, steps: stepRefs, stage: stageRef }: StoryParts): boolean {
  const story = storyRef.current;
  const stage = stageRef.current;
  const step = stepRefs.current[i];
  if (!story || !stage || !step || window.matchMedia(UNPINNED).matches) return false;
  const scroller = story.closest<HTMLElement>("[data-scroller]");
  // Sticky `top` is from the scroller's top, which the page puts at the
  // window's.
  const pinTop = (scroller?.getBoundingClientRect().top ?? 0) + (parseFloat(getComputedStyle(stage).top) || 0);
  const box = step.getBoundingClientRect();
  const top = landAt({
    top: box.top,
    height: box.height,
    vh: window.innerHeight,
    pinnedBottom: window.matchMedia(NARROW).matches ? pinTop + stage.offsetHeight : null,
    storyTop: story.getBoundingClientRect().top,
    pinTop,
    scrollTop: scroller ? scroller.scrollTop : window.scrollY,
  });
  const behavior: ScrollBehavior = reducedMotion() ? "auto" : "smooth";
  if (scroller) scroller.scrollTo({ top, behavior });
  else window.scrollTo({ top, behavior });
  return true;
}
