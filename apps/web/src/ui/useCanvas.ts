import { useEffect, useLayoutEffect, useRef, useState } from "react";

export interface CanvasSize {
  width: number;
  height: number;
  dpr: number;
}

/** Keeps a canvas's backing store at its CSS size times devicePixelRatio. */
export function useCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0, dpr: 1 });
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const dpr = globalThis.devicePixelRatio || 1;
      const width = Math.round(entry.contentRect.width * dpr);
      const height = Math.round(entry.contentRect.height * dpr);
      canvas.width = width;
      canvas.height = height;
      setSize({ width, height, dpr });
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);
  return [ref, size] as const;
}

/** Calls `draw` every animation frame while mounted. */
export function useFrame(draw: () => void) {
  const latest = useRef(draw);
  useLayoutEffect(() => {
    latest.current = draw;
  });
  useEffect(() => {
    let id = 0;
    const loop = () => {
      latest.current();
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, []);
}
