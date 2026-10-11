# OPFS ownership

Every byte of audio lives in the origin private file system. A `FileSystemSyncAccessHandle` is exclusive: while one is open on a file, any other attempt to open one on that file fails, and so does `createWritable()`. So each file has one owner at a time, and the rules below decide who.

## Layout

```
projects/<projectId>/
  project.json                       written last; its presence marks a finished import
  source.flac                        16-bit FLAC, fixed 4096-sample frames
  source.idx                         Float64 byte offset of each frame, then the file length
  derived/peaks/<layerId>.bin        min/max pyramid (engine serializePeaks)
  derived/tiles/<layerId>/<n>.bin    spectrogram tile n: 1024 columns x 256 bands, uint8
```

`models/` and `jobs/` arrive with M2.

## Who opens what

| File | Writer | When | Readers |
| --- | --- | --- | --- |
| `source.flac` | import worker, sync handle | during import only | reader worker, sync handle, while the editor is open |
| `source.idx`, `derived/**` | import worker, sync handle per file | during import only | main thread and reader worker, `getFile()` |
| `project.json` | import worker | once, after every other file is closed | main thread, `getFile()` |

Rules that follow from this:

- A folder without a valid `project.json` is an interrupted import. The project list reports these folders with their size and offers to clear them, except the one the current import is writing.
- The main thread never opens a sync handle. It reads finished files with `getFile()`, which takes no lock.
- The editor's reader worker holds `source.flac` open for as long as the project is open. Nothing may rewrite the source while the editor is open. Later milestones write new layers to new files, never over old ones.
- Closing the editor asks the reader worker to close its handle and waits for it (up to a second) before terminating the worker, so the files are free again when the project list shows.
