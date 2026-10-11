import { useState } from "react";
import type { Project } from "./project/schema";
import { Editor } from "./ui/Editor";
import { ProjectList } from "./ui/ProjectList";
import { UpdatePrompt } from "./ui/UpdatePrompt";

export function App() {
  const [open, setOpen] = useState<Project>();
  if (!globalThis.crossOriginIsolated) {
    return (
      <p className="p-4 text-red-300" data-testid="isolated">
        This page is not cross-origin isolated, so audio playback cannot start. Reload, or check the server
        headers.
      </p>
    );
  }
  return (
    <>
      {open ? (
        <Editor key={open.id} project={open} onClose={() => setOpen(undefined)} />
      ) : (
        <ProjectList onOpen={setOpen} />
      )}
      <UpdatePrompt />
    </>
  );
}
