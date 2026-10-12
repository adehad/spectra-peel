const RATE = 8000;
/** Android shows no media notification for clips under about 5 s. */
const SECONDS = 10;

function silentWav(): Blob {
  const frames = RATE * SECONDS;
  const bytes = new DataView(new ArrayBuffer(44 + frames * 2));
  const text = (at: number, s: string) => {
    for (let i = 0; i < s.length; i++) bytes.setUint8(at + i, s.charCodeAt(i));
  };
  text(0, "RIFF");
  bytes.setUint32(4, 36 + frames * 2, true);
  text(8, "WAVEfmt ");
  bytes.setUint32(16, 16, true);
  bytes.setUint16(20, 1, true);
  bytes.setUint16(22, 1, true);
  bytes.setUint32(24, RATE, true);
  bytes.setUint32(28, RATE * 2, true);
  bytes.setUint16(32, 2, true);
  bytes.setUint16(34, 16, true);
  text(36, "data");
  bytes.setUint32(40, frames * 2, true);
  return new Blob([bytes.buffer], { type: "audio/wav" });
}

/** Where a job is, for the notification's position bar. */
export interface JobTime {
  elapsedSeconds: number;
  etaSeconds: number | null;
}

function position(time: JobTime | null): void {
  try {
    if (time?.etaSeconds == null) navigator.mediaSession.setPositionState();
    else {
      const duration = time.elapsedSeconds + time.etaSeconds;
      navigator.mediaSession.setPositionState({
        duration,
        position: Math.min(time.elapsedSeconds, duration),
      });
    }
  } catch {
    // Browsers without position state throw; the title still carries the percentage.
  }
}

/**
 * Android Chrome freezes a page soon after it leaves the screen, and may kill it, unless it is
 * playing media. A silent loop keeps a long job running, and its media notification shows the job's
 * progress: the position bar runs over the elapsed time plus the estimate of what is left.
 * `start` must run inside a user gesture, or the browser refuses to play.
 */
class KeepAlive {
  private audio: HTMLAudioElement | undefined;
  private url: string | undefined;

  start(title: string, artist: string): void {
    this.url ??= URL.createObjectURL(silentWav());
    this.audio ??= Object.assign(new Audio(this.url), { loop: true });
    this.audio.play().catch(() => undefined);
    this.update(title, artist, null);
  }

  update(title: string, artist: string, time: JobTime | null): void {
    if (!this.audio || !("mediaSession" in navigator)) return;
    const session = navigator.mediaSession;
    session.metadata = new MediaMetadata({
      title,
      artist,
      artwork: [{ src: "/icons/icon-192.png", sizes: "192x192", type: "image/png" }],
    });
    position(time);
  }

  stop(): void {
    this.audio?.pause();
    this.audio = undefined;
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = undefined;
    if ("mediaSession" in navigator) navigator.mediaSession.metadata = null;
    position(null);
  }
}

export const keepAlive = new KeepAlive();
