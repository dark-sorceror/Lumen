import kit from "@/app/(landing)/page.module.css";
import IllumenMark from "@/components/IllumenMark";
import Wordmark from "@/components/Wordmark";
import { ClaudeMark } from "./Icons";

/** The sticky top bar: the logo (to the very top), three sections, then
 * the plugin and the app, both coming soon. */
export default function Nav() {
  return (
    <header className={kit.nav}>
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
          <span className={`${kit.btn} ${kit.plug} ${kit.soon}`}>
            <ClaudeMark />
            Plugin coming soon
          </span>
          <span className={`${kit.btn} ${kit.btnSolid} ${kit.soon}`}>Coming soon</span>
        </div>
      </div>
    </header>
  );
}
