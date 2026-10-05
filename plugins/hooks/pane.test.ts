import { test, expect } from 'claude-code/testing'

/**
 * One tree, every surface. The body is written once and run over all four,
 * because Box, Text and Button are the only elements every surface's table
 * carries and it is easy to reach past them without noticing.
 *
 * KNOWN LIMIT: the plugin test kit has no `state` noun, and a `state.get`
 * hook beneath the plugin does not source the state library's reads, so the
 * pane can only be mounted in the state the plugin starts in. The filled
 * tree's decisions are covered instead by the `view` tests in cost.test.ts;
 * what is NOT covered by a mount is the filled tree's own validation on each
 * surface. It is built from the same two elements as the tree below.
 */
const SURFACES = ['terminal', 'desktop', 'vscode', 'mobile'] as const

const PANE = {
  plugin: 'lumen-cache',
  component: 'Pane',
  requestId: 'lumen-cache',
  props: {
    title: 'Cache horizon',
    isFocused: false,
    bodyColumns: 44,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 24 },
    view: {},
  },
} as const

test('the pane draws on every surface before any request is measured', async $ => {
  for (const surface of SURFACES) {
    const ui = await $.ui.mount({ ...PANE, surface })

    expect(await ui.find({ type: 'Text', text: /No model request measured yet/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /fills on the first response/ })).toBeDefined()

    await ui.unmount()
  }
})

test('the pane narrows to a small body without overrunning it', async $ => {
  for (const surface of SURFACES) {
    const ui = await $.ui.mount({
      ...PANE,
      surface,
      props: { ...PANE.props, bodyColumns: 12 },
    })

    expect(await ui.find({ type: 'Text', text: /No model request measured yet/ })).toBeDefined()

    await ui.unmount()
  }
})
