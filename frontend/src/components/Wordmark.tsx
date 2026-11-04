import styles from "./Wordmark.module.css";

// The name as it's written everywhere: lowercase, in Sora.
export default function Wordmark({ className }: { className?: string }) {
  return <span className={className ? `${styles.wordmark} ${className}` : styles.wordmark}>illumen</span>;
}
