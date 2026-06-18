import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Dreamless.VAETiled",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Dreamless_VAE_Tiled") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            const self = this;

            const updateInputsAndWidgets = () => {
                const modeWidget = self.widgets.find(w => w.name === "mode");
                if (!modeWidget) return;

                const idealSize = self.computeSize();
                self.size[0] = Math.max(self.size[0], idealSize[0]);
                self.size[1] = idealSize[1];
                self.setDirtyCanvas(true, true);
            };

            setTimeout(() => { updateInputsAndWidgets(); }, 30);

            const modeWidget = self.widgets.find(w => w.name === "mode");
            if (modeWidget) {
                let originalCallback = modeWidget.callback;
                modeWidget.callback = function (value, ...args) {
                    const result = originalCallback ? originalCallback.apply(this, [value, ...args]) : value;
                    updateInputsAndWidgets();
                    return result;
                };
            }
        };
    },
});