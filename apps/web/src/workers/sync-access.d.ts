// The app compiles against lib.dom, which leaves out these worker-only OPFS members.
interface FileSystemReadWriteOptions {
  at?: number;
}

interface FileSystemSyncAccessHandle {
  read(buffer: AllowSharedBufferSource, options?: FileSystemReadWriteOptions): number;
  write(buffer: AllowSharedBufferSource, options?: FileSystemReadWriteOptions): number;
  truncate(newSize: number): void;
  getSize(): number;
  flush(): void;
  close(): void;
}

interface FileSystemFileHandle {
  /** Renames the file in place; Chrome 110 and later. */
  move(newName: string): Promise<void>;
  /** "read-only" handles can coexist with each other; the default "readwrite" is exclusive. */
  createSyncAccessHandle(options?: {
    mode?: "readwrite" | "read-only" | "readwrite-unsafe";
  }): Promise<FileSystemSyncAccessHandle>;
}
