import { app } from "../../scripts/app.js";

function hideWidget(w) {
    w.hidden = true;
    w.computeSize = () => [0, -4];
}
function showWidget(w) {
    w.hidden = false;
    w.computeSize = undefined;
}


function registerDynamicLoras(nodeName) {
    app.registerExtension({
        name: `Dreamless.DynamicLoras.${nodeName}`,
        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeData.name !== nodeName) return;

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                setTimeout(() => {
                    const countWidget = this.widgets?.find((w) => w.name === "lora_count");
                    if (!countWidget) return;

                    const updateWidgets = () => {
                        const count = Number(countWidget.value);

                        for (const w of this.widgets) {
                            const isLoraWidget =
                                w.name.startsWith("lora_air_") ||
                                w.name.startsWith("lora_name_") ||
                                w.name.startsWith("strength_model_") ||
                                w.name.startsWith("strength_clip_");

                            if (!isLoraWidget) continue;
                            const match = w.name.match(/_(\d+)$/);
                            if (!match) continue;
                            const idx = parseInt(match[1]);

                            if (idx >= count) {
                                hideWidget(w);
                            } else {
                                showWidget(w);
                            }
                        }

                        const idealSize = this.computeSize();
                        this.size[0] = Math.max(this.size[0], idealSize[0]);
                        this.size[1] = idealSize[1];
                        this.minSize = [...idealSize];
                        app.canvas.setDirty(true, true);
                    };

                    countWidget.callback = () => { updateWidgets(); };
                    updateWidgets();
                }, 100);

                return r;
            };
        }
    });
}

registerDynamicLoras("Dreamless_Multiple_LORA_Loader");
registerDynamicLoras("Dreamless_LORA_Stack");
registerDynamicLoras("Dreamless_LORA_Stack_Downloader";

app.registerExtension({
    name: "Dreamless.ImageLoaderDynamic",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if (nodeData.name !== "Dreamless_Image_Loader") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {

            const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;

            const modeWidget = this.widgets?.find(
                (w) => w.name === "mode"
            );

            if (!modeWidget) return r;

            const singleWidgets = ["image"];
            const urlWidgets = ["urls", "cache_url"];

            const allOptional = [
                ...new Set([
                    ...singleWidgets,
                    ...urlWidgets
                ])
            ];

            const updateWidgets = () => {

                const mode = modeWidget.value;

                const visible =
                    mode === "single"
                        ? singleWidgets
                        : urlWidgets;

                for (const w of this.widgets) {

                    if (!allOptional.includes(w.name)) continue;

                    if (!visible.includes(w.name)) {
                        hideWidget(w);
                    } else {
                        showWidget(w);
                    }
                }

                this.setSize(this.computeSize());

                app.canvas.setDirty(true, true);
            };

            modeWidget.callback = () => {
                updateWidgets();
            };

            requestAnimationFrame(() => updateWidgets());

            return r;
        };
    }
});

app.registerExtension({
    name: "Dreamless.NodeSizes",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const sizes = {
            "Dreamless_Loader": [420, 620],
            "Dreamless_Loader_CivitAI": [450, 800],
            "Dreamless_LORA_Stack": [380, 200],
            "Dreamless_Multiple_LORA_Loader": [380, 200],
            "Dreamless_Checkpoint_Loader": [300, 360],
            "Dreamless_LORA_Loader": [300, 360],
            "Dreamless_VAE_Loader": [360, 180],
            "Dreamless_Image_Loader": [380, 300],
            "Dreamless_Loader_CivitAI": [450, 800],
            "Dreamless_Diffusion_Model_DualCLIP_Loader_CivitAI": [450, 800],
            "Dreamless_Diffusion_Model_CLIP_Loader_CivitAI": [450, 800],
            "Dreamless_Loader_Singleton": [450, 800],
            "Dreamless_Save_Image_Advanced": [450, 560],
            "Dreamless_Smart_Latent": [350, 220],
            "Dreamless_VAE_Nodes": [280, 120],
            "Dreamless_Save_Image_Preview": [370, 540],
            "Dreamless_Load_Image": [300, 460],
            "Dreamless_KSampler_Simple" :[290, 330],
            "Dreamless_KSampler_Full" :[290, 390],
            "Dreamless_KSampler_Hires" :[340, 560],          
            "Dreamless_KSampler_Config": [310, 302],
            "Dreamless_KSampler_Config_Advanced": [300, 210],
            "Dreamless_KSampler_Advanced": [260, 160],
            "Dreamless_Preview_Image": [450, 500],
            "Dreamless_Load_Image": [280, 420],


        };

        const defaultSize = sizes[nodeData.name];

        if (!defaultSize) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;

            this.size = [...defaultSize];
            this.setSize([...defaultSize]);
            return r;
        };
    }
});

app.registerExtension({
    name: "Dreamless.KSamplerSeed",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Dreamless_KSampler" &&
            nodeData.name !== "Dreamless_KSampler_Hires") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const seedWidget = this.widgets?.find(w => w.name === "seed");
            const controlWidget = this.widgets?.find(w => w.name === "control after generate");
            if (!seedWidget || !controlWidget) return r;

            this.onExecuted = function (message) {
                const nextSeed = message?.output?.SEED?.[0] ?? message?.output?.[7]?.[0];
                if (nextSeed == null) return;
                const control = controlWidget.value;
                if (control === "fixed") return;
                if (control === "randomize") {
                    seedWidget.value = Math.floor(Math.random() * 0xFFFFFFFFFFFF);
                    return;
                }
                seedWidget.value = nextSeed;
            };

            return r;
        };
    }
});

app.registerExtension({
    name: "Dreamless.SaveImagePreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Dreamless_Save_Image_Preview") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            if (message?.images) {
                this.imgs = [];
                for (const imgData of message.images) {
                    const img = new Image();
                    img.src = `./view?filename=${encodeURIComponent(imgData.filename)}&type=${imgData.type}&subfolder=${encodeURIComponent(imgData.subfolder)}&t=${+new Date()}`;
                    this.imgs.push(img);
                }

                this.setDirtyCanvas(true, true);
            } else {
                this.imgs = null;
                this.setDirtyCanvas(true, true);
            }
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (onDrawForeground) {
                onDrawForeground.apply(this, arguments);
            }

            if (this.imgs && this.imgs.length > 0) {
                const margin = 10;
                const widgetHeight = this.widgets ? this.widgets.length * 30 : 0;
                const startY = widgetHeight + 20;

                const width = this.size[0] - (margin * 2);
                let height = this.size[1] - startY - margin;

                if (height < 60) {
                    height = width;
                    this.size[1] = startY + height + margin;
                }

                const img = this.imgs[0];
                if (img.complete) {
                    ctx.drawImage(img, margin, startY, width, height);
                }
            }
        };
    },
});

app.registerExtension({
    name: "Dreamless.ContextMirror",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Dreamless_Context") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            this.setSize([230, 300]);
        };
    },
});


const NODE_NAME = "Dreamless_Loader_Singleton";

const FIELD_DEFAULTS = {
    ckpt_air: "{model_id}@{model_version}",
    vae_air: "{model_id}@{model_version}",
    lora_air: "{model_id}@{model_version}",
    hf_api_key: "",
    civitai_api_key: "",
};

app.registerExtension({
    name: "Dreamless.RestoreDefaults",

    nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) return;

        for (const widget of node.widgets ?? []) {
            if (!(widget.name in FIELD_DEFAULTS)) continue;
            const defaultValue = FIELD_DEFAULTS[widget.name];
            const originalCallback = widget.callback;

            widget.callback = function (value) {
                if (value === null || value === undefined || String(value).trim() === "") {
                    widget.value = defaultValue;
                }

                if (originalCallback) originalCallback.call(this, widget.value);
            };
        }
    },
});

// Loaders
const LOADER_CONFIGS = {
    "Dreamless_Loader": { hasLora: true },
    "Dreamless_Loader_CivitAI": { hasLora: true },
    "Dreamless_Diffusion_Model_CLIP_Loader_CivitAI": { hasLora: false },
    "Dreamless_Diffusion_Model_DualCLIP_Loader_CivitAI": { hasLora: false },
    "Dreamless_Loader_Singleton": { hasLora: false },
};

app.registerExtension({
    name: "Dreamless.LoaderNodes",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = LOADER_CONFIGS[nodeData.name];
        if (!config) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const self = this;

            const updateVisibility = () => {
                const resolutionWidget = self.widgets.find(w => w.name === "resolution");
                const customWidthWidget = self.widgets.find(w => w.name === "custom_width");
                const customHeightWidget = self.widgets.find(w => w.name === "custom_height");

                if (resolutionWidget && customWidthWidget && customHeightWidget) {
                    const isCustom = resolutionWidget.value === "Custom (width x height)";
                    isCustom ? showWidget(customWidthWidget) : hideWidget(customWidthWidget);
                    isCustom ? showWidget(customHeightWidget) : hideWidget(customHeightWidget);
                }

                if (config.hasLora) {
                    const loraNameWidget = self.widgets.find(w => w.name === "lora_name");
                    const loraStrengthModel = self.widgets.find(w => w.name === "lora_strength_model");
                    const loraStrengthClip = self.widgets.find(w => w.name === "lora_strength_clip");

                    if (loraNameWidget && loraStrengthModel && loraStrengthClip) {
                        const hasLora = loraNameWidget.value !== "none";
                        hasLora ? showWidget(loraStrengthModel) : hideWidget(loraStrengthModel);
                        hasLora ? showWidget(loraStrengthClip) : hideWidget(loraStrengthClip);
                    }
                }

                const idealSize = self.computeSize();

                if (idealSize) {
                    idealSize[1] += 160;
                }

                self.size[0] = Math.max(self.size[0], idealSize[0]);
                self.size[1] = Math.max(self.size[1], idealSize[1]);

                app.canvas.setDirty(true, true);
            };

            const watchComboWidget = (widgetName) => {
                const widget = self.widgets.find(w => w.name === widgetName);
                if (!widget) return;

                const origCallback = widget.callback;
                widget.callback = function (value, ...args) {
                    const result = origCallback ? origCallback.apply(this, [value, ...args]) : value;
                    updateVisibility();
                    return result;
                };

                let _val = widget.value;
                Object.defineProperty(widget, "value", {
                    get() { return _val; },
                    set(v) { _val = v; updateVisibility(); },
                    configurable: true,
                });
            };

            setTimeout(() => {
                watchComboWidget("resolution");
                if (config.hasLora) watchComboWidget("lora_name");
                updateVisibility();
            }, 50);

            const spacer = this.addWidget("info", "", "", () => { });
            spacer.computeSize = () => [0, 10];
        };
    },
});
