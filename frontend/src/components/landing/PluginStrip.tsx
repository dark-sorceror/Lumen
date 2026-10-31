import kit from "@/app/(landing)/page.module.css";
import { PLUGIN_INSTALL } from "@/lib/landing";
import CopyButton from "./CopyButton";
import { ClaudeMark } from "./Icons";
import styles from "./Sections.module.css";

/** The plugin strip: the two install lines, and what the plugin shows. */
export default function PluginStrip() {
  return (
    <section className={kit.sec} id="plugin" aria-labelledby="plugin-title">
      <div className={kit.wrap}>
        <div className={styles.plugin}>
          <div>
            <span className={kit.kicker}>
              <ClaudeMark /> A plugin for Claude Code
            </span>
            <h2 className={kit.h2} id="plugin-title">
              Already in Claude Code? Add illumen to your terminal.
            </h2>
            <p className={kit.lede}>
              illumen&apos;s plugin puts a little of illumen beside your Claude Code session: how full it is, and what
              a summary dropped of what you typed, with a way to put it back.
            </p>
            <div className={kit.install}>
              {PLUGIN_INSTALL.map((line) => (
                <div className={kit.cmd} key={line}>
                  <span className={kit.cmdP}>$</span>
                  <code>{line}</code>
                  <CopyButton text={line} what="the install command" className={`${kit.btn} ${kit.btnSm}`} />
                </div>
              ))}
            </div>
          </div>
          <div>
            <ul className={styles.pkeys}>
              <li>
                <b>A line under the prompt</b>
                <span>How full it is. After a summary, how many things you typed are gone.</span>
                <code>⚠ illumen: 17% full · 2 gone · /illumen-forgot</code>
              </li>
              <li>
                <b>A row where it summarised</b>
                <span>What it dropped, in faded ink, with Copy and Put back.</span>
                <code>
                  ─ Claude Code summarised the chat · 165k → 26k ·{" "}
                  <span className={styles.tRed}>2 things you typed are gone</span> ▸
                </code>
              </li>
              <li>
                <b>/illumen-forgot</b>
                <span>
                  The same list, any time. <span className={kit.mono}>/illumen-forgot restore</span> puts them back in
                  your prompt.
                </span>
                <code>
                  <span className={styles.tFaded}>1 Never run the migrations against the shared st…</span>
                </code>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
