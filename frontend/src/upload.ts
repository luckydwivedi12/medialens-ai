export function validateImage(file: File): string | null {
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return 'Choose a JPEG, PNG or WEBP image.'
  if (!file.size || file.size > 10 * 1024 * 1024) return 'Choose a nonempty image no larger than 10 MB.'
  return null
}

export function validateVideo(file: File): string | null {
  if (!/\.(mp4|mov|avi|mkv)$/i.test(file.name)) return 'Choose an MP4, MOV, AVI or MKV video.'
  if (file.type && !['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/avi', 'video/x-matroska', 'application/octet-stream'].includes(file.type)) return 'Choose a supported video format.'
  if (!file.size || file.size > 50 * 1024 * 1024) return 'Choose a nonempty video no larger than 50 MB.'
  return null
}

export function validateDuration(duration: number): string | null {
  return !Number.isFinite(duration) || duration <= 0 || duration > 120 ? 'Choose a video no longer than 120 seconds with a valid duration.' : null
}
