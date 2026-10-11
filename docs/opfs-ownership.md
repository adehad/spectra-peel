# OPFS ownership

Every byte of audio lives in the origin private file system. A default ("readwrite") `FileSystemSyncAccessHandle` is exclusive: while one is open on a file, no other handle or `createWritable()` can open it. A "read-only" sync handle can coexist with other read-only handles. So each file has one writer, at a time when nothing reads it, and any number of readers afterwards.

## Layout

```
projects/<projectId>/
  project.json                       op log, layers and mix; written last on import
  source.flac  source.idx            16-bit FLAC in fixed 4096-sample frames, and each frame's byte offset
  layers/<layerId>.flac  .idx        FLAC layers made by ops, covering only the layer's extent
  derived/peaks/<layerId>.bin        min/max pyramid (engine serializePeaks)
  derived/tiles/<layerId>/<n>.bin    spectrogram tile n of the layer's extent: 1024 columns x 256 bands
```

A residual layer (what a denoise removed) has tiles but no audio file. Playback computes it as minuend minus subtrahend.

## Who opens what

| File | Writer | When | Readers |
| --- | --- | --- | --- |
| `source.flac`, `layers/*.flac` | import or denoise worker, readwrite sync handle | while the file is new | reader worker and denoise worker, read-only sync handles |
| `*.idx`, `derived/**` | the same worker, a readwrite sync handle per file | while the file is new | main thread and workers, `getFile()` |
| `project.json` | import worker on import; afterwards the main thread, `createWritable()` | after every file it lists is closed | main thread, `getFile()` |

Rules that follow from this:

- A folder without a valid `project.json` is an interrupted import. The project list reports these folders with their size and offers to clear them, except the one the current import is writing.
- An op's files are listed in `project.json` only once the worker has closed them. A cancelled or failed op deletes its files, and nothing ever lists them.
- Readers use read-only handles, so playback carries on while a denoise reads the same source.
- The main thread never opens a sync handle. It reads finished files with `getFile()`, which takes no lock.
- Undo writes `project.json` first, then deletes the op's files once the player that still reads them has been replaced.
- Closing the editor asks the reader worker to close its handles and waits for it (up to a second) before terminating the worker, so the files are free again when the project list shows.
