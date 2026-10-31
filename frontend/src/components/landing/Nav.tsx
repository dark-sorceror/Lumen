import kit from "@/app/(landing)/page.module.css";
import IllumenMark from "@/components/IllumenMark";
import Wordmark from "@/components/Wordmark";
import { ClaudeMark } from "./Icons";

/** The sticky top bar: the logo (to the very top), three sections, the
 * plugin, and Get started. */
export default function Nav({ onStart }: { onStart: (opener: HTMLElement) => void }) {
  return (
    <header className={kit.nav} data-behind-sheet>
      <div className={`${kit.wrap} ${kit.navIn}`}>
        <a className={kit.brand} href="#top">
          <IllumenMark size={24} />
          <Wordmark className={kit.brandWord} />
        </a>
        <nav className={kit.links} aria-label="Sections">
          <a href="#forgets">How it works</a>
          <a href="#ways-in">Ways in</a>
          <a href="#plugin">Plugin</a>
        </nav>
        <div className={kit.navAct}>
          <a className={`${kit.btn} ${kit.plug}`} href="#plugin">
            <ClaudeMark />
            Get the plugin
          </a>
          <a
            className={`${kit.btn} ${kit.btnSolid}`}
            href="#start"
            onClick={(e) => {
              e.preventDefault();
              onStart(e.currentTarget);
            }}
          >
            Get started
          </a>
        </div>
      </div>
    </header>
  );
}
