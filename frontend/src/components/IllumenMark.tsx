type Props = {
  size?: number;
  className?: string;
};

// illumen's mark: a disc in the text colour with a lit rim
// (--mark-lit) cut apart by a thin gap. Decorative: the wordmark beside it
// names it. app/icon.svg is the same mark on a warm-black tile.
export default function IllumenMark({ size = 22, className }: Props) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <path d="M12.51 21.79A22 22 0 1 1 42.21 51.49A21 21 0 1 0 12.51 21.79Z" fill="var(--mark-lit)" />
      <path d="M10.07 30.32A18.5 18.5 0 1 1 33.68 53.93A22 22 0 0 1 10.07 30.32Z" fill="currentColor" />
    </svg>
  );
}
