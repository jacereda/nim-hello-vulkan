import cv, logging, sequtils, glm
from vk import nil, toDeviceSize
{.push warning[ObservableStores]: off.}

type
  UniformBufferObject = object
    mvp: Mat4[float32]
  Vertex = tuple[
    x: float32,
    y: float32,
    z: float32,
    r: float32,
    g: float32,
    b: float32,
    a: float32,
    ]

let acs: ptr vk.AllocationCallbacks = nil
const MAX_FRAMES_IN_FLIGHT = 2
var
  inst: vk.Instance
  dev: vk.Device
  pdev: vk.PhysicalDevice
  dum: vk.DebugUtilsMessengerEXT
  surf: vk.SurfaceKHR
  schain: vk.SwapchainKHR
  iviews: seq[vk.ImageView]
  vbuffer: vk.Buffer
  vmem: vk.DeviceMemory
  ibuffer: vk.Buffer
  imem: vk.DeviceMemory
  dpool: vk.DescriptorPool
  fbuffers: seq[vk.Framebuffer]
  ubuffers: seq[vk.Buffer]
  umems: seq[vk.DeviceMemory]
  layout: vk.PipelineLayout
  rpass: vk.RenderPass
  dslayout: vk.DescriptorSetLayout
  dsets: seq[vk.DescriptorSet]
  pline: vk.Pipeline
  cpool: vk.CommandPool
  cbuffers: seq[vk.CommandBuffer]
  gqueue: vk.Queue
  pqueue: vk.Queue
  savailable: array[MAX_FRAMES_IN_FLIGHT,vk.Semaphore]
  sfinished: array[MAX_FRAMES_IN_FLIGHT,vk.Semaphore]
  fcompleted: array[MAX_FRAMES_IN_FLIGHT,vk.Fence]
  frame: uint
  proj: Mat4[float32]
  pos: Vec3[float32] = vec3(0.0f,0.0f,-7.0f)

addHandler(newConsoleLogger())

proc chk(e: vk.Result) =
  if e != vk.SUCCESS:
     fatal(e.repr)
     assert(false)

proc newInstance(): vk.Instance =
  let ai = vk.mkApplicationInfo(
    pApplicationName = cast[ptr char]("NimGL Vulkan Example".cstring),
    applicationVersion = vk.MAKE_VERSION(1, 0, 0),
    pEngineName = cast[ptr char]("No Engine".cstring),
    engineVersion = vk.MAKE_VERSION(1, 0, 0),
    apiVersion = vk.MAKE_VERSION(1,2,0)
  )
  let extensions = [
    vk.EXT_DEBUG_UTILS_EXTENSION_NAME,
    vk.KHR_SURFACE_EXTENSION_NAME,
    vk.KHR_WIN32_SURFACE_EXTENSION_NAME,
  ]
  let layers = [
    "VK_LAYER_KHRONOS_validation".cstring,
    "VK_LAYER_LUNARG_standard_validation".cstring,
    ]
  let ici = vk.mkInstanceCreateInfo(
    pApplicationInfo = ai.unsafeAddr,
    enabledExtensionCount = vk.ulen(extensions),
    ppEnabledExtensionNames = cast[ptr ptr char](extensions[0].unsafeAddr),
    enabledLayerCount = vk.ulen(layers),
    ppEnabledLayerNames = cast[ptr ptr char](layers[0].unsafeAddr),
  )
  chk vk.CreateInstance(ici.unsafeAddr, acs, result.addr)
  var ep: array[256,vk.ExtensionProperties]
  var num = vk.ulen(ep)
  discard vk.EnumerateInstanceExtensionProperties(nil, num.addr, ep[0].addr)

proc pickPhysicalDevice(inst: vk.Instance, index: int): vk.PhysicalDevice =
  var devs: array[32, vk.PhysicalDevice]
  var ndevs = vk.ulen(devs)
  chk vk.EnumeratePhysicalDevices(inst, ndevs.addr, devs[0].addr)
  devs[index]

func gfxFamily(fp: vk.QueueFamilyProperties, pres: bool): bool =
  vk.QUEUE_GRAPHICS in fp.queueFlags

func presentFamily(fp: vk.QueueFamilyProperties, pres: bool): bool =
  pres

proc familyIndex(filt: proc(fp: vk.QueueFamilyProperties, pres: bool):bool): uint32 =
  var qfs: array[32,vk.QueueFamilyProperties]
  var nqfs = vk.ulen(qfs)
  vk.GetPhysicalDeviceQueueFamilyProperties(pdev, nqfs.addr, qfs[0].addr)
  for i in 0..<nqfs:
    let qf = qfs[i]
    var pres: vk.Bool32
    chk vk.GetPhysicalDeviceSurfaceSupportKHR(pdev, i, surf, pres.addr)
    if filt(qf, pres.bool != vk.FALSE.bool):
        return i.uint32
  0xffffffffu32

proc newDevice(): vk.Device =
  let qp = 0.0.float32
  let dqci = [
    vk.mkDeviceQueueCreateInfo(queueFamilyIndex = familyIndex(gfxFamily),
                              queueCount = 1u32,
                              pQueuePriorities = qp.unsafeAddr),
    ]
  let ext = [
    vk.KHR_SWAPCHAIN_EXTENSION_NAME,
    ]
  let dci = vk.mkDeviceCreateInfo(queueCreateInfoCount = vk.ulen(dqci),
                                  pQueueCreateInfos = dqci[0].unsafeAddr,
                                  enabledLayerCount=0,
                                  ppEnabledLayerNames=nil,
                                  enabledExtensionCount=vk.ulen(ext),
                                  ppEnabledExtensionNames=cast[ptr ptr char](ext[0].unsafeAddr),
                                  pEnabledFeatures=nil)
  chk vk.CreateDevice(pdev, dci.unsafeAddr, acs, result.addr)

proc debugCB(messageSeverity: vk.DebugUtilsMessageSeverityFlagBitsEXT,
             messageTypes: vk.DebugUtilsMessageTypeFlagsEXT,
             pCallbackData: ptr vk.DebugUtilsMessengerCallbackDataEXT,
             pUserData: pointer,
            ): vk.Bool32 {.cdecl.} =
    echo "validation error: " & $pCallbackData.pMessage
    vk.TRUE

proc newDebugMessenger(): vk.DebugUtilsMessengerEXT =
  let dumci = vk.mkDebugUtilsMessengerCreateInfoEXT(
    messageSeverity = { vk.DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_EXT,
                        vk.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_EXT,
                        vk.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_EXT },
    messageType = { vk.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_EXT,
                    vk.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_EXT,
                    vk.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_EXT },
    pfnUserCallback = debugCB,
  )
  chk vk.CreateDebugUtilsMessengerEXT(inst, dumci.unsafeAddr, acs, result.addr)

proc newSurface(win: pointer, hinst: pointer): vk.SurfaceKHR =
  let sci = vk.mkWin32SurfaceCreateInfoKHR(
    hinstance = hinst,
    hwnd = win,
  )
  chk vk.CreateWin32SurfaceKHR(inst, sci.unsafeAddr, acs, result.addr)

proc surfaceFormat(): vk.SurfaceFormatKHR =
  var sf: array[256,vk.SurfaceFormatKHR]
  var nsf = vk.ulen(sf)
  chk vk.GetPhysicalDeviceSurfaceFormatsKHR(pdev, surf, nsf.addr, sf[0].addr)
  sf[0]


proc newSwapchain(): vk.SwapchainKHR =
  var sc: vk.SurfaceCapabilitiesKHR
  chk vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(pdev, surf, sc.addr)
  let sf = surfaceFormat()
  let qfi = [
    familyIndex(gfxFamily),
    familyIndex(presentFamily),
  ]
  let scci = vk.mkSwapchainCreateInfoKHR(
    surface = surf,
    minImageCount = sc.minImageCount,
    imageFormat = sf.format,
    imageColorSpace = sf.colorSpace,
    imageExtent = sc.currentExtent,
    imageArrayLayers = 1,
    imageUsage = {vk.IMAGE_USAGE_COLOR_ATTACHMENT},
    imageSharingMode = if qfi[0] == qfi[1]: vk.SHARING_MODE_EXCLUSIVE else: vk.SHARING_MODE_CONCURRENT,
    queueFamilyIndexCount = vk.ulen(qfi),
    pQueueFamilyIndices = qfi[0].unsafeAddr,
    preTransform = vk.SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    compositeAlpha = vk.COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    presentMode = vk.PRESENT_MODE_FIFO_KHR,
    clipped = vk.TRUE,
  )
  chk vk.CreateSwapchainKHR(dev, scci.unsafeAddr, acs, result.addr)

proc createImageViews(): seq[vk.ImageView] =
  var sci : array[256,vk.Image]
  var nsci = vk.ulen(sci)
  chk vk.GetSwapchainImagesKHR(dev, schain, nsci.addr, nil)
  chk vk.GetSwapchainImagesKHR(dev, schain, nsci.addr, sci[0].addr)
  result = newSeq[vk.ImageView]()
  for i in 0..<nsci:
    let ivci = vk.mkImageViewCreateInfo(
      image = sci[i],
      viewType = vk.IMAGE_VIEW_TYPE_2D,
      format = surfaceFormat().format,
      components = vk.mkComponentMapping(
        r = vk.COMPONENT_SWIZZLE_IDENTITY,
        g = vk.COMPONENT_SWIZZLE_IDENTITY,
        b = vk.COMPONENT_SWIZZLE_IDENTITY,
        a = vk.COMPONENT_SWIZZLE_IDENTITY,
      ),
      subresourceRange = vk.mkImageSubresourceRange(
        aspectMask = {vk.IMAGE_ASPECT_COLOR},
        baseMipLevel = 0,
        levelCount = 1,
        baseArrayLayer = 0,
        layerCount = 1,
      ),
    )
    var iv: vk.ImageView
    chk vk.CreateImageView(dev, ivci.unsafeAddr, acs, iv.addr)
    result.add(iv)

proc newShaderModule(code: string): vk.ShaderModule =
  let smci = vk.mkShaderModuleCreateInfo(
    codeSize = code.len.uint,
    pCode = cast[ptr uint32](code[0].unsafeAddr),
  )
  chk vk.CreateShaderModule(dev, smci.unsafeAddr, acs, result.addr)

proc newDescriptorSetLayout(): vk.DescriptorSetLayout =
  let dslbs = [
    vk.mkDescriptorSetLayoutBinding(
      binding = 0,
      descriptorType = vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount = 1,
      stageFlags = {vk.SHADER_STAGE_VERTEX},
    ),
  ]
  let dslci = vk.mkDescriptorSetLayoutCreateInfo(
    bindingCount = vk.ulen(dslbs),
    pBindings = dslbs[0].unsafeAddr,
    )
  chk vk.CreateDescriptorSetLayout(dev, dslci.unsafeAddr, acs, result.addr)

proc newPipelineLayout(): vk.PipelineLayout =
  let pl = vk.mkPipelineLayoutCreateInfo(
    setLayoutCount = 1,
    pSetLayouts = dslayout.addr,
    pPushConstantRanges = nil,
  )
  chk vk.CreatePipelineLayout(dev, pl.unsafeAddr, acs, result.addr)

proc newRenderPass(): vk.RenderPass =
  let sf = surfaceFormat()
  let ads = [
    vk.mkAttachmentDescription(
      format = sf.format,
      samples = vk.SAMPLE_COUNT_1_BIT,
      loadOp = vk.ATTACHMENT_LOAD_OP_CLEAR,
      storeOp = vk.ATTACHMENT_STORE_OP_STORE,
      stencilLoadOp = vk.ATTACHMENT_LOAD_OP_DONT_CARE,
      stencilStoreOp = vk.ATTACHMENT_STORE_OP_DONT_CARE,
      initialLayout = vk.IMAGE_LAYOUT_UNDEFINED,
      finalLayout = vk.IMAGE_LAYOUT_PRESENT_SRC_KHR,
    ),
  ]
  let ats = [
    vk.mkAttachmentReference(
      attachment = 0,
      layout = vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    ),
  ]
  let sds = [
    vk.mkSubpassDescription(
      pipelineBindPoint = vk.PIPELINE_BIND_POINT_GRAPHICS,
      pInputAttachments = nil,
      colorAttachmentCount = vk.ulen(ats),
      pColorAttachments = ats[0].unsafeAddr,
      pPreserveAttachments = nil,
    ),
  ]
  let sdes = [
    vk.mkSubpassDependency(
      srcSubPass = vk.SUBPASS_EXTERNAL,
      dstSubpass = 0,
      srcStageMask = {vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT},
      dstStageMask = {vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT},
      dstAccessMask = {vk.ACCESS_COLOR_ATTACHMENT_WRITE},
    )
  ]
  let rpci = vk.mkRenderPassCreateInfo(
    attachmentCount = vk.ulen(ads),
    pAttachments = ads[0].unsafeAddr,
    subpassCount = vk.ulen(sds),
    pSubpasses = sds[0].unsafeAddr,
    dependencyCount = vk.ulen(sdes),
    pDependencies = sdes[0].unsafeAddr,
  )
  chk vk.CreateRenderPass(dev, rpci.unsafeAddr, acs, result.addr)

proc newGraphicsPipeline(): vk.Pipeline =
  let vibds = [
    vk.mkVertexInputBindingDescription(
      binding = 0,
      stride = Vertex.sizeOf.uint32,
      inputRate = vk.VERTEX_INPUT_RATE_VERTEX,
    ),
  ]
  let viads = [
    vk.mkVertexInputAttributeDescription(
      location = 0,
      binding = 0,
      format = vk.FORMAT_R32G32B32_SFLOAT,
      offset = offsetOf(Vertex, x).uint32,
    ),
    vk.mkVertexInputAttributeDescription(
      location = 1,
      binding = 0,
      format = vk.FORMAT_R32G32B32A32_SFLOAT,
      offset = offsetOf(Vertex, r).uint32,
    ),
  ]
  let pvisci = vk.mkPipelineVertexInputStateCreateInfo(
    vertexBindingDescriptionCount = vk.ulen(vibds),
    pVertexBindingDescriptions = vibds[0].unsafeAddr,
    vertexAttributeDescriptionCount = vk.ulen(viads),
    pVertexAttributeDescriptions = viads[0].unsafeAddr,
  )
  let piasci = vk.mkPipelineInputAssemblyStateCreateInfo(
    topology = vk.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    primitiveRestartEnable = vk.FALSE,
  )
  let vps = [
    vk.mkViewport(
      x = 0.0f,
      y = 0.0f,
      width = canvasWidth().float32,
      height = canvasHeight().float32,
      minDepth = 0.0f,
      maxDepth = 1.0f,
    ),
  ]
  let scs = [
    vk.mkRect2D(
      offset = vk.mkOffset2D(
        x = 0,
        y = 0,
      ),
      extent = vk.mkExtent2D(
        width = canvasWidth(),
        height = canvasHeight(),
      ),
    ),
  ]
  let pvsci = vk.mkPipelineViewportStateCreateInfo(
    viewportCount = vk.ulen(vps),
    pViewports = vps[0].unsafeAddr,
    scissorCount = vk.ulen(scs),
    pScissors = scs[0].unsafeAddr,
    )
  let prsci = vk.mkPipelineRasterizationStateCreateInfo(
    depthClampEnable = vk.FALSE,
    rasterizerDiscardEnable = vk.FALSE,
    polygonMode = vk.POLYGON_MODE_FILL,
    cullMode = {vk.CULL_MODE_BACK},
    frontFace = vk.FRONT_FACE_CLOCKWISE,
    depthBiasEnable = vk.FALSE,
    depthBiasConstantFactor = 0.0f,
    depthBiasClamp = 0.0f,
    depthBiasSlopeFactor = 0.0f,
    lineWidth = 1.0f,
  )
  let pmsci = vk.mkPipelineMultisampleStateCreateInfo(
    rasterizationSamples = vk.SAMPLE_COUNT_1_BIT,
    sampleShadingEnable = vk.FALSE,
    minSampleShading = 1.0f,
    alphaToCoverageEnable = vk.FALSE,
    alphaToOneEnable = vk.FALSE,
  )
  let pcbass = [
    vk.mkPipelineColorBlendAttachmentState(
      blendEnable = vk.FALSE,
      srcColorBlendFactor = vk.BLEND_FACTOR_ONE,
      dstColorBlendFactor = vk.BLEND_FACTOR_ZERO,
      colorBlendOp = vk.BLEND_OP_ADD,
      srcAlphaBlendFactor = vk.BLEND_FACTOR_ONE,
      dstAlphaBlendFactor = vk.BLEND_FACTOR_ZERO,
      alphaBlendOp = vk.BLEND_OP_ADD,
      colorWriteMask = {vk.COLOR_COMPONENT_R, vk.COLOR_COMPONENT_G, vk.COLOR_COMPONENT_B, vk.COLOR_COMPONENT_A}
    ),
  ]
  let pcbsci = vk.mkPipelineColorBlendStateCreateInfo(
    logicOpEnable = vk.FALSE,
    logicOp = vk.LOGIC_OP_COPY,
    attachmentCount = vk.ulen(pcbass),
    pAttachments = pcbass[0].unsafeAddr,
    blendConstants = [0.0f, 0.0f, 0.0f, 0.0f],
  )
  let ds = [
    vk.DYNAMIC_STATE_VIEWPORT,
  ]
  let pdsci = vk.mkPipelineDynamicStateCreateInfo(
    dynamicStateCount = vk.ulen(ds),
    pDynamicStates = ds[0].unsafeAddr,
  )

  const vcode = slurp("vert.spv")
  let vmod = newShaderModule(vcode)
  defer: vk.DestroyShaderModule(dev, vmod, acs)
  const fcode = slurp("frag.spv")
  let fmod = newShaderModule(fcode)
  defer: vk.DestroyShaderModule(dev, fmod, acs)
  let psscis =  [
    vk.mkPipelineShaderStageCreateInfo(
      stage = vk.SHADER_STAGE_VERTEX_BIT,
      module = vmod,
      pName = "main".cstring,
    ),
    vk.mkPipelineShaderStageCreateInfo(
      stage = vk.SHADER_STAGE_FRAGMENT_BIT,
      module = fmod,
      pName = "main".cstring,
    ),
  ]

  let gpcis = [
    vk.mkGraphicsPipelineCreateInfo(
      stageCount = vk.ulen(psscis),
      pStages = psscis[0].unsafeAddr,
      pVertexInputState = pvisci.unsafeAddr,
      pInputAssemblyState = piasci.unsafeAddr,
      pViewportState = pvsci.unsafeAddr,
      pRasterizationState = prsci.unsafeAddr,
      pMultisampleState = pmsci.unsafeAddr,
      pColorBlendState = pcbsci.unsafeAddr,
      pDynamicState = pdsci.unsafeAddr,
      layout = layout,
      renderPass = rpass,
      subpass = 0,
      basePipelineIndex = -1,
    ),
  ]
  chk vk.CreateGraphicsPipelines(dev,
    pipelineCache = nil,
    createInfoCount = vk.ulen(gpcis),
    pCreateInfos = gpcis[0].unsafeAddr,
    pAllocator = acs,
    pPipelines = result.addr)

proc createFramebuffers(): seq[vk.Framebuffer] =
  result = newSeq[vk.Framebuffer]()
  for i in 0..iviews.high:
    let fbci = vk.mkFramebufferCreateInfo(
      renderPass = rpass,
      attachmentCount = 1,
      pAttachments = iviews[i].addr,
      width = canvasWidth(),
      height = canvasHeight(),
      layers = 1,
      )
    var fb: vk.FrameBuffer
    chk vk.CreateFramebuffer(dev, fbci.unsafeAddr, acs, fb.addr)
    result.add(fb)

proc findMemoryType(bits: uint32, properties: vk.MemoryPropertyFlags): uint32 =
  var mp: vk.PhysicalDeviceMemoryProperties
  vk.GetPhysicalDeviceMemoryProperties(pdev, mp.addr)
  for i in 0..<mp.memoryTypeCount:
    if ((bits and (1u32 shl i)) != 0) and (properties < mp.memoryTypes[i].propertyFlags):
      return i
  assert(false)
  (uint32.high)

proc newBuffer(usage: vk.BufferUsageFlags, size: int): vk.Buffer =
  let bci = vk.mkBufferCreateInfo(
    size = size,
    usage = usage,
    sharingMode = vk.SHARING_MODE_EXCLUSIVE,
    pQueueFamilyIndices = nil,
  )
  chk vk.CreateBuffer(dev, bci.unsafeAddr, acs, result.addr)

proc newDevBuffer(usage: vk.BufferUsageFlags, size: int): vk.Buffer =
  newBuffer(usage + {vk.BUFFER_USAGE_TRANSFER_DST}, size)

proc newStaBuffer(size: int): vk.Buffer =
  newBuffer({vk.BUFFER_USAGE_TRANSFER_SRC}, size)


proc allocBufferMemory(b: vk.Buffer, flags: vk.MemoryPropertyFlags): vk.DeviceMemory =
    var mr: vk.MemoryRequirements
    vk.GetBufferMemoryRequirements(dev, b, mr.addr)
    let mai = vk.mkMemoryAllocateInfo(
      allocationSize = mr.size,
      memoryTypeIndex = findMemoryType(mr.memoryTypeBits, flags),
      )
    chk vk.AllocateMemory(dev, mai.unsafeAddr, acs, result.addr)
    chk vk.BindBufferMemory(dev, b, result, 0)

proc allocDevBufferMemory(b: vk.Buffer): vk.DeviceMemory =
  allocBufferMemory(b, {vk.MEMORY_PROPERTY_DEVICE_LOCAL})

proc allocStaBufferMemory(b: vk.Buffer): vk.DeviceMemory =
  allocBufferMemory(b, {vk.MEMORY_PROPERTY_HOST_VISIBLE, vk.MEMORY_PROPERTY_HOST_COHERENT})

template withMappedMemory(dm: vk.DeviceMemory, nm: untyped, t: untyped, ops: untyped) =
  var nm: ptr t
  chk vk.MapMemory(dev, dm, 0, t.sizeOf, {}, nm.addr)
  ops
  vk.UnmapMemory(dev, dm)


proc copyBuffer(sb: vk.Buffer, db: vk.Buffer, sz: int) =
  let cbai = vk.mkCommandBufferAllocateInfo(
    level = vk.COMMAND_BUFFER_LEVEL_PRIMARY,
    commandPool = cpool,
    commandBufferCount = 1,
  )
  var cb: vk.CommandBuffer
  chk vk.AllocateCommandBuffers(dev, cbai.unsafeAddr, cb.addr)
  let cbbi = vk.mkCommandBufferBeginInfo(
    flags = {vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT},
  )
  chk vk.BeginCommandBuffer(cb, cbbi.unsafeAddr)
  let bc = vk.mkBufferCopy(
    srcOffset = 0,
    dstOffset = 0,
    size = sz,
  )
  vk.CmdCopyBuffer(cb, sb, db, 1, bc.unsafeAddr)
  chk vk.EndCommandBuffer(cb)
  let si = vk.mkSubmitInfo(
    pWaitSemaphores = nil,
    pWaitDstStageMask = nil,
    commandBufferCount = 1,
    pCommandBuffers = cb.unsafeAddr,
    pSignalSemaphores = nil,
  )
  chk vk.QueueSubmit(gqueue, 1, si.unsafeAddr, nil)
  chk vk.QueueWaitIdle(gqueue)
  vk.FreeCommandBuffers(dev, cpool, 1, cb.unsafeAddr)

template withStagingMemory(b: vk.Buffer, dm: vk.DeviceMemory, nm: untyped, t: untyped, ops: untyped) =
  let sb = newStaBuffer(t.sizeOf)
  let sbm = allocStaBufferMemory(sb)
  var nm: ptr t
  chk vk.MapMemory(dev, sbm, 0, t.sizeOf, {}, nm.addr)
  ops
  vk.UnmapMemory(dev, sbm)
  copyBuffer(sb, b, t.sizeOf)
  vk.FreeMemory(dev, sbm, acs)
  vk.DestroyBuffer(dev, sb, acs)

proc createUniformBuffers(): seq[vk.Buffer] =
  result = newSeq[vk.Buffer]()
  for i in 0..iviews.high:
    let ub = newBuffer({vk.BUFFER_USAGE_UNIFORM_BUFFER}, UniformBufferObject.sizeOf)
    result.add(ub)

proc createUniformBuffersMemory(): seq[vk.DeviceMemory] =
  result = newSeq[vk.DeviceMemory]()
  for ub in ubuffers:
    var mr: vk.MemoryRequirements
    vk.GetBufferMemoryRequirements(dev, ub, mr.addr)
    let mai = vk.mkMemoryAllocateInfo(
      allocationSize = mr.size,
      memoryTypeIndex = findMemoryType(mr.memoryTypeBits, {vk.MEMORY_PROPERTY_HOST_VISIBLE, vk.MEMORY_PROPERTY_HOST_COHERENT}),
      )
    var dm: vk.DeviceMemory
    chk vk.AllocateMemory(dev, mai.unsafeAddr, acs, dm.addr)
    chk vk.BindBufferMemory(dev, ub, dm, 0)
    result.add(dm)

proc newDescriptorPool(): vk.DescriptorPool =
  let dpss = [
    vk.mkDescriptorPoolSize(
      `type` = vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount = vk.ulen(iviews),
    ),
  ]
  let dpci = vk.mkDescriptorPoolCreateInfo(
    poolSizeCount = vk.ulen(dpss),
    pPoolSizes = dpss[0].unsafeAddr,
    maxSets = vk.ulen(iviews),
    )
  chk vk.CreateDescriptorPool(dev, dpci.unsafeAddr, acs, result.addr)

proc createDescriptorSets(): seq[vk.DescriptorSet] =
  result = newSeq[vk.DescriptorSet](iviews.len)
  var dsls = dslayout.repeat(result.len)
  let dsai = vk.mkDescriptorSetAllocateInfo(
    descriptorPool = dpool,
    descriptorSetCount = vk.ulen(result),
    pSetLayouts = dsls[0].addr,
    )
  chk vk.AllocateDescriptorSets(dev, dsai.unsafeAddr, result[0].addr)
  for i in 0..result.high:
    let dbi = vk.mkDescriptorBufferInfo(
      buffer = ubuffers[i],
      offset = 0,
      range = UniformBufferObject.sizeOf,
      )
    let wds = vk.mkWriteDescriptorSet(
      dstSet = result[i],
      dstBinding = 0u32,
      dstArrayElement = 0u32,
      descriptorType = vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount = 1,
      pBufferInfo = dbi.unsafeAddr,
      pImageInfo = nil,
      pTexelBufferView = nil,
      )
    vk.UpdateDescriptorSets(dev, 1u32, wds.unsafeAddr, 0u32, nil)


proc newCommandPool(): vk.CommandPool =
  let cpci = vk.mkCommandPoolCreateInfo(
    queueFamilyIndex = familyIndex(gfxFamily),
    )
  chk vk.CreateCommandPool(dev, cpci.unsafeAddr, acs, result.addr)

proc createCommandBuffers(): seq[vk.CommandBuffer] =
  result = newSeq[vk.CommandBuffer](fbuffers.len)
  let cbai = vk.mkCommandBufferAllocateInfo(
    commandPool = cpool,
    level = vk.COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount = vk.ulen(result),
    )
  chk vk.AllocateCommandBuffers(dev, cbai.unsafeAddr, result[0].addr)

proc renderPass() =
  let cv = vk.ClearValue(
    color: vk.ClearColorValue(
      `float32`: [0.0f, 0.0f, 0.0f, 1.0f]
      )
  )
  for i in 0..fbuffers.high:
    let cbuffer = cbuffers[i]
    let cbbi = vk.mkCommandBufferBeginInfo()
    chk vk.BeginCommandBuffer(cbuffer, cbbi.unsafeAddr)
    let rpbi = vk.mkRenderPassBeginInfo(
      renderPass = rpass,
      framebuffer = fbuffers[i],
      renderArea =  vk.mkRect2D(
        offset = vk.mkOffset2D(
          x = 0,
          y = 0,
        ),
        extent = vk.mkExtent2D(
          width = canvasWidth(),
          height = canvasHeight(),
        ),
      ),
      clearValueCount = 1,
      pClearValues = cv.unsafeAddr,
    )
    let vp = vk.mkViewport(
      x = 0.0f,
      y = 0.0f,
      width = canvasWidth().float32,
      height = canvasHeight().float32,
      minDepth = 0.0f,
      maxDepth = 1.0f,
    )
    vk.CmdSetViewport(cbuffer, 0u32, 1u32, vp.unsafeAddr)
    vk.CmdBeginRenderPass(cbuffer, rpbi.unsafeAddr, vk.SUBPASS_CONTENTS_INLINE)
    vk.CmdBindPipeline(cbuffer, vk.PIPELINE_BIND_POINT_GRAPHICS, pline)
    let vbuffers = [ vbuffer, ]
    let offsets = [ toDeviceSize(0), ]
    vk.CmdBindVertexBuffers(cbuffer, 0.uint32, vk.ulen(vbuffers), vbuffers[0].unsafeAddr, offsets[0].unsafeAddr)
    vk.CmdBindIndexBuffer(cbuffer, ibuffer, 0, vk.INDEX_TYPE_UINT32)
    vk.CmdBindDescriptorSets(cbuffer, vk.PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, dsets[i].addr, 0, nil)
    vk.CmdDrawIndexed(cbuffer, 36u32, 1u32, 0u32, 0i32, 0u32)
    vk.CmdEndRenderPass(cbuffer)
    chk vk.EndCommandBuffer(cbuffer)

proc newSemaphore(): vk.Semaphore =
  let sci = vk.mkSemaphoreCreateInfo()
  chk vk.CreateSemaphore(dev, sci.unsafeAddr, acs, result.addr)

proc createSemaphores(): array[MAX_FRAMES_IN_FLIGHT,vk.Semaphore] =
  for i in 0..<MAX_FRAMES_IN_FLIGHT:
    result[i] = newSemaphore()

proc newFence(): vk.Fence =
  let fci = vk.mkFenceCreateInfo(
    flags = {vk.FENCE_CREATE_SIGNALED},
  )
  chk vk.CreateFence(dev, fci.unsafeAddr, acs, result.addr)

proc createFences(): array[MAX_FRAMES_IN_FLIGHT, vk.Fence] =
  for i in 0..<MAX_FRAMES_IN_FLIGHT:
    result[i] = newFence()

proc updateUniformBuffers(idx: uint32) =
  let view =  mat4f()
  let model = mat4f()
  .translate(pos)
  .rotate(-2.0 * PI * mouseY().int.toFloat / canvasHeight().int.toFloat, 1, 0, 0)
  .rotate(-2.0 * PI * mouseX().int.toFloat / canvasWidth().int.toFloat, 0, 1, 0)
  withMappedMemory(umems[idx], ubo, UniformBufferObject):
    ubo[] = UniformBufferObject(
      mvp: proj * view * model,
    )

proc cleanupSwapChain() =
  chk vk.DeviceWaitIdle(dev, )
  for fbuffer in fbuffers:
    vk.DestroyFramebuffer(dev, fbuffer, acs)
  fbuffers.delete(0, fbuffers.high)
  vk.FreeCommandBuffers(dev, cpool, vk.ulen(cbuffers), cbuffers[0].addr)
  cbuffers.delete(0, cbuffers.high)
  vk.DestroyDescriptorPool(dev, dpool, acs)
  vk.DestroyPipeline(dev, pline, acs)
  vk.DestroyPipelineLayout(dev, layout, acs)
  vk.DestroyRenderPass(dev, rpass, acs)
  for iview in iviews:
    vk.DestroyImageView(dev, iview, acs)
  iviews.delete(0, iviews.high)
  vk.DestroySwapchainKHR(dev, schain, acs)
  for ubuffer in ubuffers:
    vk.DestroyBuffer(dev, ubuffer, acs)
  ubuffers.delete(0, ubuffers.high)
  for umem in umems:
    vk.FreeMemory(dev, umem, acs)
  umems.delete(0, umems.high)

proc createSwapChain() =
  schain = newSwapchain()
  iviews = createImageViews()
  layout = newPipelineLayout()
  rpass = newRenderPass()
  pline = newGraphicsPipeline()
  fbuffers = createFramebuffers()
  ubuffers = createUniformBuffers()
  umems = createUniformBuffersMemory()
  dpool = newDescriptorPool()
  dsets = createDescriptorSets()
  cbuffers = createCommandBuffers()
  renderPass()

proc recreateSwapChain() =
  cleanupSwapChain()
  createSwapChain()

proc initVulkan(win: pointer, hinst: pointer) =
  vk.loadProcs(nil)
  inst = newInstance()
  vk.loadProcs(inst)
  dum = newDebugMessenger()
  pdev = inst.pickPhysicalDevice(0)
  surf = newSurface(win, hinst)
  dev = newDevice()
  vk.GetDeviceQueue(dev, familyIndex(gfxFamily), 0, gqueue.addr)
  vk.GetDeviceQueue(dev, familyIndex(presentFamily), 0, pqueue.addr)
  dslayout = newDescriptorSetLayout()
  vbuffer = newDevBuffer({vk.BUFFER_USAGE_VERTEX_BUFFER}, (array[8, Vertex]).sizeOf)
  vmem = allocDevBufferMemory(vbuffer)
  ibuffer = newDevBuffer({vk.BUFFER_USAGE_INDEX_BUFFER}, (array[36, uint32]).sizeOf)
  imem = allocDevBufferMemory(ibuffer)
  savailable = createSemaphores()
  sfinished = createSemaphores()
  fcompleted = createFences()
  cpool = newCommandPool()
  createSwapChain()
  withStagingMemory(vbuffer, vmem, vert, array[8, Vertex]):
    for i in 0..<8:
      vert[i].x = if (i and 1) != 0: 1.0f else: -1.0f
      vert[i].y = if (i and 2) != 0: 1.0f else: -1.0f
      vert[i].z = if (i and 4) != 0: 1.0f else: -1.0f
      vert[i].r = if (i and 1) != 0: 1.0f else: 0.0f
      vert[i].g = if (i and 2) != 0: 1.0f else: 0.0f
      vert[i].b = if (i and 4) != 0: 1.0f else: 0.0f
      vert[i].a = 1.0f
  withStagingMemory(ibuffer, imem, ind, array[36, uint32]):
    ind[] = [
      4'u32, 5'u32, 7'u32, 7'u32, 6'u32, 4'u32, # front
      1'u32, 0'u32, 2'u32, 2'u32, 3'u32, 1'u32, # back
      0'u32, 4'u32, 6'u32, 6'u32, 2'u32, 0'u32, # left
      5'u32, 1'u32, 3'u32, 3'u32, 7'u32, 5'u32, # right
      6'u32, 7'u32, 3'u32, 3'u32, 2'u32, 6'u32, # top
      0'u32, 1'u32, 5'u32, 5'u32, 4'u32, 0'u32, # bottom
    ]

proc termVulkan() =
  cleanupSwapChain()
  for fence in fcompleted:
    vk.DestroyFence(dev, fence, acs)
  for sem in sfinished:
    vk.DestroySemaphore(dev, sem, acs)
  for sem in savailable:
    vk.DestroySemaphore(dev, sem, acs)
  vk.DestroyCommandPool(dev, cpool, acs)
  vk.FreeMemory(dev, imem, acs)
  vk.DestroyBuffer(dev, ibuffer, acs)
  vk.FreeMemory(dev, vmem, acs)
  vk.DestroyBuffer(dev, vbuffer, acs)
  vk.DestroyDescriptorSetLayout(dev, dslayout, acs)
  vk.DestroySurfaceKHR(inst, surf, acs)
  vk.DestroyDevice(dev, acs)
  vk.DestroyDebugUtilsMessengerEXT(inst, dum, acs)
  vk.DestroyInstance(inst, acs)


proc draw() =
  let curr = frame mod MAX_FRAMES_IN_FLIGHT
  chk vk.WaitForFences(dev, 1, fcompleted[curr].addr, vk.TRUE, uint64.high)
  chk vk.ResetFences(dev, 1, fcompleted[curr].addr)
  var idx : uint32
  chk vk.AcquireNextImageKHR(dev, schain, uint.high, savailable[curr], nil, idx.addr)
  updateUniformBuffers(idx)
  let ws = [{vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT},]
  let si = vk.mkSubmitInfo(
    waitSemaphoreCount = 1,
    pWaitSemaphores = savailable[curr].addr,
    pWaitDstStageMask = ws[0].unsafeAddr,
    commandBufferCount = 1,
    pCommandBuffers = cbuffers[idx].addr,
    signalSemaphoreCount = 1,
    pSignalSemaphores = sfinished[curr].addr,
  )
  chk vk.QueueSubmit(gqueue, 1, si.unsafeAddr, fcompleted[curr])
  let pi = vk.mkPresentInfoKHR(
    waitSemaphoreCount = 1,
    pWaitSemaphores = sfinished[curr].addr,
    swapchainCount = 1,
    pSwapchains = schain.addr,
    pImageIndices = idx.addr,
    )
  chk vk.QueuePresentKHR(pqueue, pi.unsafeAddr)
  frame.inc


proc update() =
  if keyPressed cvkLeftArrow: pos.x -= 0.1f
  if keyPressed cvkRightArrow: pos.x += 0.1f
  if keyPressed cvkDownArrow: pos.y += 0.1f
  if keyPressed cvkUpArrow: pos.y -= 0.1f
  if keyPressed cvkX: pos.z -= 0.1f
  if keyPressed cvkZ: pos.z += 0.1f
  draw()

proc resized(w: uint, h: uint) =
  recreateSwapChain()
  proj = perspective(45.0f, w.float / h.float, 0.1f, 100000.0f)

proc handler(e: ev): int {.cdecl.} =
  case e.eventType:
    of cveResize:
      resized(e.eventWidth, e.eventHeight)
    of cveGLInit:
      initVulkan(cast[pointer](e.eventArg0), cast[pointer](e.eventArg1))
    of cveGLTerm:
      termVulkan()
    of cveUpdate:
      update()
      return 1
    else:
      return 0

discard run handler
{.pop.}
