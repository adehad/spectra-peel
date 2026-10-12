import { useEffect, useState } from "react";
import type { Project } from "./project/schema";
import { projectDir, readProject } from "./storage/opfs";
import { Editor } from "./ui/Editor";
import { ProjectList } from "./ui/ProjectList";
import { UpdatePrompt } from "./ui/UpdatePrompt";

/** The open project's id is kept in the URL hash, so a reload, or a phone killing the page, reopens it. */
const hashId = () => decodeURIComponent(location.hash.slice(1));

export function App() {
  const [open, setOpen] = useState<Project>();
  const [restoring, setRestoring] = useState(() => hashId() !== "");
  const show = (project: Project | undefined) => {
    history.replaceState(null, "", project ? `#${encodeURIComponent(project.id)}` : location.pathname);
    setOpen(project);
  };

  useEffect(() => {
    const id = hashId();
    if (!id) return;
    projectDir(id)
      .then(readProject)
      .then(setOpen, () => history.replaceState(null, "", location.pathname))
      .finally(() => setRestoring(false));
  }, []);

  if (!globalThis.crossOriginIsolated) {
    return (
      <p className="p-4 text-red-300" data-testid="isolated">
        This page is not cross-origin isolated, so audio playback cannot start. Reload, or check the server
        headers.
      </p>
    );
  }
  if (restoring) return <p className="p-4 text-dim">Opening…</p>;
  return (
    <>
      {open ? (
        <Editor key={open.id} project={open} onClose={() => show(undefined)} />
      ) : (
        <ProjectList onOpen={show} />
      )}
      <UpdatePrompt />
    </>
  );
}
