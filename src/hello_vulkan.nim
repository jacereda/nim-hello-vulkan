import vk, cv, logging, sequtils, glm
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

let acs: ptr VkAllocationCallbacks = nil
const MAX_FRAMES_IN_FLIGHT = 2
var
  inst: VkInstance
  dev: VkDevice
  pdev: VkPhysicalDevice
  dum: VkDebugUtilsMessengerEXT
  surf: VkSurfaceKHR
  schain: VkSwapchainKHR
  iviews: seq[VkImageView]
  vbuffer: VkBuffer
  vmem: VkDeviceMemory
  ibuffer: VkBuffer
  imem: VkDeviceMemory
  dpool: VkDescriptorPool
  fbuffers: seq[VkFramebuffer]
  ubuffers: seq[VkBuffer]
  umems: seq[VkDeviceMemory]
  layout: VkPipelineLayout
  rpass: VkRenderPass
  dslayout: VkDescriptorSetLayout
  dsets: seq[VkDescriptorSet]
  pline: VkPipeline
  cpool: VkCommandPool
  cbuffers: seq[VkCommandBuffer]
  gqueue: VkQueue
  pqueue: VkQueue
  available: array[MAX_FRAMES_IN_FLIGHT,VkSemaphore]
  finished: array[MAX_FRAMES_IN_FLIGHT,VkSemaphore]
  completed: array[MAX_FRAMES_IN_FLIGHT,VkFence]
  frame: uint
  proj: Mat4[float32]
  pos: Vec3[float32] = vec3(0.0f,0.0f,-7.0f)

addHandler(newConsoleLogger())

func errStr(e: VkResult): string =
  case e
  of VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS"
  of VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT"
  of VK_ERROR_NOT_PERMITTED_EXT: "VK_ERROR_NOT_PERMITTED_EXT"
  of VK_ERROR_FRAGMENTATION: "VK_ERROR_FRAGMENTATION"
  of VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT"
  of VK_ERROR_INCOMPATIBLE_VERSION_KHR: "VK_ERROR_INCOMPATIBLE_VERSION_KHR"
  of VK_ERROR_INVALID_EXTERNAL_HANDLE: "VK_ERROR_INVALID_EXTERNAL_HANDLE"
  of VK_ERROR_OUT_OF_POOL_MEMORY: "VK_ERROR_OUT_OF_POOL_MEMORY"
  of VK_ERROR_INVALID_SHADER_NV: "VK_ERROR_INVALID_SHADER_NV"
  of VK_ERROR_VALIDATION_FAILED_EXT: "VK_ERROR_VALIDATION_FAILED_EXT"
  of VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR"
  of VK_ERROR_OUT_OF_DATE_KHR: "VK_ERROR_OUT_OF_DATE_KHR"
  of VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR"
  of VK_ERROR_SURFACE_LOST_KHR: "VK_ERROR_SURFACE_LOST_KHR"
  of VK_ERROR_UNKNOWN: "VK_ERROR_UNKNOWN"
  of VK_ERROR_FRAGMENTED_POOL: "VK_ERROR_FRAGMENTED_POOL"
  of VK_ERROR_FORMAT_NOT_SUPPORTED: "VK_ERROR_FORMAT_NOT_SUPPORTED"
  of VK_ERROR_TOO_MANY_OBJECTS: "VK_ERROR_TOO_MANY_OBJECTS"
  of VK_ERROR_INCOMPATIBLE_DRIVER: "VK_ERROR_INCOMPATIBLE_DRIVER"
  of VK_ERROR_FEATURE_NOT_PRESENT: "VK_ERROR_FEATURE_NOT_PRESENT"
  of VK_ERROR_EXTENSION_NOT_PRESENT: "VK_ERROR_EXTENSION_NOT_PRESENT"
  of VK_ERROR_LAYER_NOT_PRESENT: "VK_ERROR_LAYER_NOT_PRESENT"
  of VK_ERROR_MEMORY_MAP_FAILED: "VK_ERROR_MEMORY_MAP_FAILED"
  of VK_ERROR_DEVICE_LOST: "VK_ERROR_DEVICE_LOST"
  of VK_ERROR_INITIALIZATION_FAILED: "VK_ERROR_INITIALIZATION_FAILED"
  of VK_ERROR_OUT_OF_DEVICE_MEMORY: "VK_ERROR_OUT_OF_DEVICE_MEMORY"
  of VK_ERROR_OUT_OF_HOST_MEMORY: "VK_ERROR_OUT_OF_HOST_MEMORY"
  of VK_NOT_READY: "VK_NOT_READY"
  of VK_TIMEOUT: "VK_TIMEOUT"
  of VK_EVENT_SET: "VK_EVENT_SET"
  of VK_EVENT_RESET: "VK_EVENT_RESET"
  of VK_INCOMPLETE: "VK_INCOMPLETE"
  of VK_SUBOPTIMAL_KHR: "VK_SUBOPTIMAL_KHR"
  of VK_THREAD_IDLE_KHR: "VK_THREAD_IDLE_KHR"
  of VK_THREAD_DONE_KHR: "VK_THREAD_DONE_KHR"
  of VK_OPERATION_DEFERRED_KHR: "VK_OPERATION_DEFERRED_KHR"
  of VK_OPERATION_NOT_DEFERRED_KHR: "VK_OPERATION_NOT_DEFERRED_KHR"
  of VK_PIPELINE_COMPILE_REQUIRED_EXT: "VK_PIPELINE_COMPILE_REQUIRED_EXT"
  of VK_SUCCESS: "OK"

proc chk(e: VkResult) =
  if e != VK_SUCCESS:
     fatal(errStr(e))
     assert(false)

proc newVkInstance(): VkInstance =
  let ai = mkVkApplicationInfo(
    pApplicationName = cast[ptr char]("NimGL Vulkan Example".cstring),
    applicationVersion = VK_MAKE_VERSION(1, 0, 0),
    pEngineName = cast[ptr char]("No Engine".cstring),
    engineVersion = VK_MAKE_VERSION(1, 0, 0),
    apiVersion = VK_MAKE_VERSION(1,2,0)
  )
  let extensions = [
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
  ]
  let layers = [
    "VK_LAYER_KHRONOS_validation".cstring,
    "VK_LAYER_LUNARG_standard_validation".cstring,
    ]
  let ici = mkVkInstanceCreateInfo(
    pApplicationInfo = ai.unsafeAddr,
    enabledExtensionCount = extensions.len.uint32,
    ppEnabledExtensionNames = cast[ptr ptr char](extensions[0].unsafeAddr),
    enabledLayerCount = layers.len.uint32,
    ppEnabledLayerNames = cast[ptr ptr char](layers[0].unsafeAddr),
  )
  chk vkCreateInstance(ici.unsafeAddr, acs, result.addr)
  var ep: array[256,VkExtensionProperties]
  var num = ep.len.uint32
  discard vkEnumerateInstanceExtensionProperties(nil, num.addr, ep[0].addr)

proc pickPhysicalDevice(inst: VkInstance, index: int): VkPhysicalDevice =
  var devs: array[32, VkPhysicalDevice]
  var ndevs = devs.len.uint32
  chk inst.vkEnumeratePhysicalDevices(ndevs.addr, devs[0].addr)
  return devs[index]

func gfxFamily(fp: VkQueueFamilyProperties, pres: bool): bool =
  return (fp.queueFlags.int and VK_QUEUE_GRAPHICS_BIT.int) != 0

func presentFamily(fp: VkQueueFamilyProperties, pres: bool): bool =
  return pres

proc familyIndex(filt: proc(fp: VkQueueFamilyProperties, pres: bool):bool): uint32 =
  var qfs: array[32,VkQueueFamilyProperties]
  var nqfs = qfs.len.uint32
  pdev.vkGetPhysicalDeviceQueueFamilyProperties(nqfs.addr, qfs[0].addr)
  for i in 0..<nqfs:
    let qf = qfs[i]
    var pres: VkBool32
    chk pdev.vkGetPhysicalDeviceSurfaceSupportKHR(i, surf, pres.addr)
    if filt(qf, pres.bool != VK_FALSE.bool):
        return i.uint32
  return 0xffffffffu32

proc newDevice(): VkDevice =
  let qp = 0.0.float32
  let dqci = [
    mkVkDeviceQueueCreateInfo(queueFamilyIndex = familyIndex(gfxFamily),
                              queueCount = 1u32,
                              pQueuePriorities = qp.unsafeAddr),
    ]
  let ext = [
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    ]
  let dci = mkVkDeviceCreateInfo(queueCreateInfoCount = dqci.len.uint32,
                                  pQueueCreateInfos = dqci[0].unsafeAddr,
                                  enabledLayerCount=0,
                                  ppEnabledLayerNames=nil,
                                  enabledExtensionCount=ext.len.uint32,
                                  ppEnabledExtensionNames=cast[ptr ptr char](ext[0].unsafeAddr),
                                  pEnabledFeatures=nil)
  chk pdev.vkCreateDevice(dci.unsafeAddr, acs, result.addr)

proc debugCB(messageSeverity: VkDebugUtilsMessageSeverityFlagBitsEXT,
             messageTypes: VkDebugUtilsMessageTypeFlagsEXT,
             pCallbackData: ptr VkDebugUtilsMessengerCallbackDataEXT,
             pUserData: pointer,
            ): VkBool32 {.cdecl.} =
    echo "validation error: " & $pCallbackData.pMessage
    VK_TRUE

proc newDebugMessenger(): VkDebugUtilsMessengerEXT =
  let dumci = mkVkDebugUtilsMessengerCreateInfoEXT(
    messageSeverity = (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT.uint32 or
                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT.uint32 or
                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT.uint32).VkDebugUtilsMessageSeverityFlagsEXT,
    messageType = (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT.uint32 or
                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT.uint32 or
                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT.uint32).VkDebugUtilsMessageTypeFlagsEXT,
    pfnUserCallback = debugCB,
  )
  chk inst.vkCreateDebugUtilsMessengerEXT(dumci.unsafeAddr, acs, result.addr)

proc newSurface(win: pointer, hinst: pointer): VkSurfaceKHR =
  let sci = mkVkWin32SurfaceCreateInfoKHR(
    hinstance = hinst,
    hwnd = win,
  )
  chk inst.vkCreateWin32SurfaceKHR(sci.unsafeAddr, acs, result.addr)

proc surfaceFormat(): VkSurfaceFormatKHR =
  var sf: array[256,VkSurfaceFormatKHR]
  var nsf = sf.len.uint32
  chk pdev.vkGetPhysicalDeviceSurfaceFormatsKHR(surf, nsf.addr, sf[0].addr)
  return sf[0]


proc newSwapchain(): VkSwapchainKHR =
  var sc: VkSurfaceCapabilitiesKHR
  chk pdev.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(surf, sc.addr)
  let sf = surfaceFormat()
  let qfi = [
    familyIndex(gfxFamily),
    familyIndex(presentFamily),
  ]
  let scci = mkVkSwapchainCreateInfoKHR(
    surface = surf,
    minImageCount = sc.minImageCount,
    imageFormat = sf.format,
    imageColorSpace = sf.colorSpace,
    imageExtent = sc.currentExtent,
    imageArrayLayers = 1,
    imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT.VkImageUsageFlags,
    imageSharingMode = if qfi[0] == qfi[1]: VK_SHARING_MODE_EXCLUSIVE else: VK_SHARING_MODE_CONCURRENT,
    queueFamilyIndexCount = qfi.len.uint32,
    pQueueFamilyIndices = qfi[0].unsafeAddr,
    preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    presentMode = VK_PRESENT_MODE_FIFO_KHR,
    clipped = VK_TRUE,
  )
  chk dev.vkCreateSwapchainKHR(scci.unsafeAddr, acs, result.addr)

proc createImageViews(): seq[VkImageView] =
  var sci : array[256,VkImage]
  var nsci = sci.len.uint32
  chk dev.vkGetSwapchainImagesKHR(schain, nsci.addr, nil)
  chk dev.vkGetSwapchainImagesKHR(schain, nsci.addr, sci[0].addr)
  result = newSeq[VkImageView]()
  for i in 0..<nsci:
    let ivci = mkVkImageViewCreateInfo(
      image = sci[i],
      viewType = VK_IMAGE_VIEW_TYPE_2D,
      format = surfaceFormat().format,
      components = mkVkComponentMapping(
        r = VK_COMPONENT_SWIZZLE_IDENTITY,
        g = VK_COMPONENT_SWIZZLE_IDENTITY,
        b = VK_COMPONENT_SWIZZLE_IDENTITY,
        a = VK_COMPONENT_SWIZZLE_IDENTITY,
      ),
      subresourceRange = mkVkImageSubresourceRange(
        aspectMask = VK_IMAGE_ASPECT_COLOR_BIT.VkImageAspectFlags,
        baseMipLevel = 0,
        levelCount = 1,
        baseArrayLayer = 0,
        layerCount = 1,
      ),
    )
    var iv: VkImageView
    chk dev.vkCreateImageView(ivci.unsafeAddr, acs, iv.addr)
    result.add(iv)

proc newShaderModule(code: string): VkShaderModule =
  let smci = mkVkShaderModuleCreateInfo(
    codeSize = code.len.uint,
    pCode = cast[ptr uint32](code[0].unsafeAddr),
  )
  chk dev.vkCreateShaderModule(smci.unsafeAddr, acs, result.addr)

proc newDescriptorSetLayout(): VkDescriptorSetLayout =
  let dslbs = [
    mkVkDescriptorSetLayoutBinding(
      binding = 0,
      descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount = 1,
      stageFlags = VK_SHADER_STAGE_VERTEX_BIT.VkShaderStageFlags,
    ),
  ]
  let dslci = mkVkDescriptorSetLayoutCreateInfo(
    bindingCount = dslbs.len.uint32,
    pBindings = dslbs[0].unsafeAddr,
    )
  chk dev.vkCreateDescriptorSetLayout(dslci.unsafeAddr, acs, result.addr)

proc newPipelineLayout(): VkPipelineLayout =
  let pl = mkVkPipelineLayoutCreateInfo(
    setLayoutCount = 1,
    pSetLayouts = dslayout.addr,
    pPushConstantRanges = nil,
  )
  chk dev.vkCreatePipelineLayout(pl.unsafeAddr, acs, result.addr)

proc newRenderPass(): VkRenderPass =
  let sf = surfaceFormat()
  let ads = [
    mkVkAttachmentDescription(
      format = sf.format,
      samples = VK_SAMPLE_COUNT_1_BIT,
      loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    ),
  ]
  let ats = [
    mkVkAttachmentReference(
      attachment = 0,
      layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    ),
  ]
  let sds = [
    mkVkSubpassDescription(
      pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      pInputAttachments = nil,
      colorAttachmentCount = ats.len.uint32,
      pColorAttachments = ats[0].unsafeAddr,
      pPreserveAttachments = nil,
    ),
  ]
  let sdes = [
    mkVkSubpassDependency(
      srcSubPass = VK_SUBPASS_EXTERNAL,
      dstSubpass = 0,
      srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT.VkPipelineStageFlags,
      dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT.VkPipelineStageFlags,
      dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT.VkAccessFlags,
    )
  ]
  let rpci = mkVkRenderPassCreateInfo(
    attachmentCount = ads.len.uint32,
    pAttachments = ads[0].unsafeAddr,
    subpassCount = sds.len.uint32,
    pSubpasses = sds[0].unsafeAddr,
    dependencyCount = sdes.len.uint32,
    pDependencies = sdes[0].unsafeAddr,
  )
  chk dev.vkCreateRenderPass(rpci.unsafeAddr, acs, result.addr)

proc newGraphicsPipeline(): VkPipeline =
  let vibds = [
    mkVkVertexInputBindingDescription(
      binding = 0,
      stride = Vertex.sizeOf.uint32,
      inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    ),
  ]
  let viads = [
    mkVkVertexInputAttributeDescription(
      location = 0,
      binding = 0,
      format = VK_FORMAT_R32G32B32_SFLOAT,
      offset = offsetOf(Vertex, x).uint32,
    ),
    mkVkVertexInputAttributeDescription(
      location = 1,
      binding = 0,
      format = VK_FORMAT_R32G32B32A32_SFLOAT,
      offset = offsetOf(Vertex, r).uint32,
    ),
  ]
  let pvisci = mkVkPipelineVertexInputStateCreateInfo(
    vertexBindingDescriptionCount = vibds.len.uint32,
    pVertexBindingDescriptions = vibds[0].unsafeAddr,
    vertexAttributeDescriptionCount = viads.len.uint32,
    pVertexAttributeDescriptions = viads[0].unsafeAddr,
  )
  let piasci = mkVkPipelineInputAssemblyStateCreateInfo(
    topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    primitiveRestartEnable = VK_FALSE,
  )
  let vps = [
    mkVkViewport(
      x = 0.0f,
      y = 0.0f,
      width = cvWidth().float32,
      height = cvHeight().float32,
      minDepth = 0.0f,
      maxDepth = 1.0f,
    ),
  ]
  let scs = [
    mkVkRect2D(
      offset = mkVkOffset2D(
        x = 0,
        y = 0,
      ),
      extent = mkVkExtent2D(
        width = cvWidth(),
        height = cvHeight(),
      ),
    ),
  ]
  let pvsci = mkVkPipelineViewportStateCreateInfo(
    viewportCount = vps.len.uint32,
    pViewports = vps[0].unsafeAddr,
    scissorCount = scs.len.uint32,
    pScissors = scs[0].unsafeAddr,
    )
  let prsci = mkVkPipelineRasterizationStateCreateInfo(
    depthClampEnable = VK_FALSE,
    rasterizerDiscardEnable = VK_FALSE,
    polygonMode = VK_POLYGON_MODE_FILL,
    cullMode = VK_CULL_MODE_BACK_BIT.VkCullModeFlags,
    frontFace = VK_FRONT_FACE_CLOCKWISE,
    depthBiasEnable = VK_FALSE,
    depthBiasConstantFactor = 0.0f,
    depthBiasClamp = 0.0f,
    depthBiasSlopeFactor = 0.0f,
    lineWidth = 1.0f,
  )
  let pmsci = mkVkPipelineMultisampleStateCreateInfo(
    rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    sampleShadingEnable = VK_FALSE,
    minSampleShading = 1.0f,
    alphaToCoverageEnable = VK_FALSE,
    alphaToOneEnable = VK_FALSE,
  )
  let pcbass = [
    mkVkPipelineColorBlendAttachmentState(
      blendEnable = VK_FALSE,
      srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      colorBlendOp = VK_BLEND_OP_ADD,
      srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      alphaBlendOp = VK_BLEND_OP_ADD,
      colorWriteMask = (VK_COLOR_COMPONENT_R_BIT.uint or VK_COLOR_COMPONENT_G_BIT.uint or VK_COLOR_COMPONENT_B_BIT.uint or VK_COLOR_COMPONENT_A_BIT.uint).VkColorComponentFlags
    ),
  ]
  let pcbsci = mkVkPipelineColorBlendStateCreateInfo(
    logicOpEnable = VK_FALSE,
    logicOp = VK_LOGIC_OP_COPY,
    attachmentCount = pcbass.len.uint32,
    pAttachments = pcbass[0].unsafeAddr,
    blendConstants = [0.0f, 0.0f, 0.0f, 0.0f],
  )
  let ds = [
    VK_DYNAMIC_STATE_VIEWPORT,
  ]
  let pdsci = mkVkPipelineDynamicStateCreateInfo(
    dynamicStateCount = ds.len.uint32,
    pDynamicStates = ds[0].unsafeAddr,
  )

  const vcode = slurp("vert.spv")
  let vmod = newShaderModule(vcode)
  defer: dev.vkDestroyShaderModule(vmod, acs)
  const fcode = slurp("frag.spv")
  let fmod = newShaderModule(fcode)
  defer: dev.vkDestroyShaderModule(fmod, acs)
  let psscis =  [
    mkVkPipelineShaderStageCreateInfo(
      stage = VK_SHADER_STAGE_VERTEX_BIT,
      module = vmod,
      pName = "main".cstring,
    ),
    mkVkPipelineShaderStageCreateInfo(
      stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      module = fmod,
      pName = "main".cstring,
    ),
  ]

  let gpcis = [
    mkVkGraphicsPipelineCreateInfo(
      stageCount = psscis.len.uint32,
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
  chk dev.vkCreateGraphicsPipelines(
    pipelineCache = nil,
    createInfoCount = gpcis.len.uint32,
    pCreateInfos = gpcis[0].unsafeAddr,
    pAllocator = acs,
    pPipelines = result.addr)

proc createFramebuffers(): seq[VkFramebuffer] =
  result = newSeq[VkFramebuffer]()
  for i in 0..iviews.high:
    let fbci = mkVkFramebufferCreateInfo(
      renderPass = rpass,
      attachmentCount = 1,
      pAttachments = iviews[i].addr,
      width = cvWidth(),
      height = cvHeight(),
      layers = 1,
      )
    var fb: VkFrameBuffer
    chk dev.vkCreateFramebuffer(fbci.unsafeAddr, acs, fb.addr)
    result.add(fb)

proc findMemoryType(bits: uint32, properties: VkMemoryPropertyFlags): uint32 =
  var mp: VkPhysicalDeviceMemoryProperties
  pdev.vkGetPhysicalDeviceMemoryProperties(mp.addr)
  for i in 0..<mp.memoryTypeCount:
    if ((bits and (1u32 shl i)) != 0) and ((mp.memoryTypes[i].propertyFlags.uint32 and properties.uint32) == properties.uint32):
      return i
  assert(false)
  return (uint32.high)

proc newBuffer(usage: VkBufferUsageFlags, size: int): VkBuffer =
  let bci = mkVkBufferCreateInfo(
    size = size.VkDeviceSize,
    usage = usage,
    sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    pQueueFamilyIndices = nil,
  )
  chk dev.vkCreateBuffer(bci.unsafeAddr, acs, result.addr)

proc newDevBuffer(usage: VkBufferUsageFlags, size: int): VkBuffer =
  newBuffer((usage.uint or VK_BUFFER_USAGE_TRANSFER_DST_BIT.uint).VkBufferUsageFlags, size)

proc newStaBuffer(size: int): VkBuffer =
  newBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT.VkBufferUsageFlags, size)


proc allocBufferMemory(b: VkBuffer, flags: VkMemoryPropertyFlags): VkDeviceMemory =
    var mr: VkMemoryRequirements
    dev.vkGetBufferMemoryRequirements(b, mr.addr)
    let mai = mkVkMemoryAllocateInfo(
      allocationSize = mr.size,
      memoryTypeIndex = findMemoryType(mr.memoryTypeBits, flags),
      )
    chk dev.vkAllocateMemory(mai.unsafeAddr, acs, result.addr)
    chk dev.vkBindBufferMemory(b, result, 0.VkDeviceSize)

proc allocDevBufferMemory(b: VkBuffer): VkDeviceMemory =
  allocBufferMemory(b, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.VkMemoryPropertyFlags)

proc allocStaBufferMemory(b: VkBuffer): VkDeviceMemory =
  allocBufferMemory(b, (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.uint32 or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT.uint32).VkMemoryPropertyFlags)

template withMappedMemory(dm: VkDeviceMemory, nm: untyped, t: untyped, ops: untyped) =
  var nm: ptr t
  chk dev.vkMapMemory(dm, 0.VkDeviceSize, t.sizeOf.VkDeviceSize, 0.VkMemoryMapFlags, nm.addr)
  ops
  dev.vkUnmapMemory(dm)


proc copyBuffer(sb: VkBuffer, db: VkBuffer, sz: int) =
  let cbai = mkVkCommandBufferAllocateInfo(
    level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandPool = cpool,
    commandBufferCount = 1,
  )
  var cb: VkCommandBuffer
  chk dev.vkAllocateCommandBuffers(cbai.unsafeAddr, cb.addr)
  let cbbi = mkVkCommandBufferBeginInfo(
    flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT.VkCommandBufferUsageFlags,
  )
  chk cb.vkBeginCommandBuffer(cbbi.unsafeAddr)
  let bc = mkVkBufferCopy(
    srcOffset = 0.VkDeviceSize,
    dstOffset = 0.VkDeviceSize,
    size = sz.VkDeviceSize,
  )
  cb.vkCmdCopyBuffer(sb, db, 1, bc.unsafeAddr)
  chk cb.vkEndCommandBuffer()
  let si = mkVkSubmitInfo(
    pWaitSemaphores = nil,
    pWaitDstStageMask = nil,
    commandBufferCount = 1,
    pCommandBuffers = cb.unsafeAddr,
    pSignalSemaphores = nil,
  )
  chk gqueue.vkQueueSubmit(1, si.unsafeAddr, nil)
  chk gqueue.vkQueueWaitIdle()
  dev.vkFreeCommandBuffers(cpool, 1, cb.unsafeAddr)

template withStagingMemory(b: VkBuffer, dm: VkDeviceMemory, nm: untyped, t: untyped, ops: untyped) =
  let sb = newStaBuffer(t.sizeOf)
  let sbm = allocStaBufferMemory(sb)
  var nm: ptr t
  chk dev.vkMapMemory(sbm, 0.VkDeviceSize, t.sizeOf.VkDeviceSize, 0.VkMemoryMapFlags, nm.addr)
  ops
  dev.vkUnmapMemory(sbm)
  copyBuffer(sb, b, t.sizeOf)
  dev.vkFreeMemory(sbm, acs)
  dev.vkDestroyBuffer(sb, acs)

proc createUniformBuffers(): seq[VkBuffer] =
  result = newSeq[VkBuffer]()
  for i in 0..iviews.high:
    let ub = newBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT.VkBufferUsageFlags, UniformBufferObject.sizeOf)
    result.add(ub)

proc createUniformBuffersMemory(): seq[VkDeviceMemory] =
  result = newSeq[VkDeviceMemory]()
  for ub in ubuffers:
    var mr: VkMemoryRequirements
    dev.vkGetBufferMemoryRequirements(ub, mr.addr)
    let mai = mkVkMemoryAllocateInfo(
      allocationSize = mr.size,
      memoryTypeIndex = findMemoryType(mr.memoryTypeBits, (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.uint32 or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT.uint32).VkMemoryPropertyFlags),
      )
    var dm: VkDeviceMemory
    chk dev.vkAllocateMemory(mai.unsafeAddr, acs, dm.addr)
    chk dev.vkBindBufferMemory(ub, dm, 0.VkDeviceSize)
    result.add(dm)

proc newDescriptorPool(): VkDescriptorPool =
  let dpss = [
    mkVkDescriptorPoolSize(
      `type` = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount = iviews.len.uint32,
    ),
  ]
  let dpci = mkVkDescriptorPoolCreateInfo(
    poolSizeCount = dpss.len.uint32,
    pPoolSizes = dpss[0].unsafeAddr,
    maxSets = iviews.len.uint32,
    )
  chk dev.vkCreateDescriptorPool(dpci.unsafeAddr, acs, result.addr)

proc createDescriptorSets(): seq[VkDescriptorSet] =
  result = newSeq[VkDescriptorSet](iviews.len)
  var dsls = dslayout.repeat(result.len)
  let dsai = mkVkDescriptorSetAllocateInfo(
    descriptorPool = dpool,
    descriptorSetCount = result.len.uint32,
    pSetLayouts = dsls[0].addr,
    )
  chk dev.vkAllocateDescriptorSets(dsai.unsafeAddr, result[0].addr)
  for i in 0..result.high:
    let dbi = mkVkDescriptorBufferInfo(
      buffer = ubuffers[i],
      offset = 0.VkDeviceSize,
      range = UniformBufferObject.sizeOf.VkDeviceSize,
      )
    let wds = mkVkWriteDescriptorSet(
      dstSet = result[i],
      dstBinding = 0u32,
      dstArrayElement = 0u32,
      descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount = 1,
      pBufferInfo = dbi.unsafeAddr,
      pImageInfo = nil,
      pTexelBufferView = nil,
      )
    dev.vkUpdateDescriptorSets(1u32, wds.unsafeAddr, 0u32, nil)


proc newCommandPool(): VkCommandPool =
  let cpci = mkVkCommandPoolCreateInfo(
    queueFamilyIndex = familyIndex(gfxFamily),
    )
  chk dev.vkCreateCommandPool(cpci.unsafeAddr, acs, result.addr)

proc createCommandBuffers(): seq[VkCommandBuffer] =
  result = newSeq[VkCommandBuffer](fbuffers.len)
  let cbai = mkVkCommandBufferAllocateInfo(
    commandPool = cpool,
    level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount = result.len.uint32,
    )
  chk dev.vkAllocateCommandBuffers(cbai.unsafeAddr, result[0].addr)

proc renderPass() =
  let cv = VkClearValue(
    color: VkClearColorValue(
      `float32`: [0.0f, 0.0f, 0.0f, 1.0f]
      )
  )
  for i in 0..fbuffers.high:
    let cbuffer = cbuffers[i]
    let cbbi = mkVkCommandBufferBeginInfo()
    chk cbuffer.vkBeginCommandBuffer(cbbi.unsafeAddr)
    let rpbi = mkVkRenderPassBeginInfo(
      renderPass = rpass,
      framebuffer = fbuffers[i],
      renderArea =  mkVkRect2D(
        offset = mkVkOffset2D(
          x = 0,
          y = 0,
        ),
        extent = mkVkExtent2D(
          width = cvWidth(),
          height = cvHeight(),
        ),
      ),
      clearValueCount = 1,
      pClearValues = cv.unsafeAddr,
    )
    let vp = mkVkViewport(
      x = 0.0f,
      y = 0.0f,
      width = cvWidth().float32,
      height = cvHeight().float32,
      minDepth = 0.0f,
      maxDepth = 1.0f,
    )
    cbuffer.vkCmdSetViewport(0u32, 1u32, vp.unsafeAddr)
    cbuffer.vkCmdBeginRenderPass(rpbi.unsafeAddr, VK_SUBPASS_CONTENTS_INLINE)
    cbuffer.vkCmdBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pline)
    let vbuffers = [ vbuffer, ]
    let offsets = [ 0.VkDeviceSize, ]
    cbuffer.vkCmdBindVertexBuffers(0.uint32, vbuffers.len.uint32, vbuffers[0].unsafeAddr, offsets[0].unsafeAddr)
    cbuffer.vkCmdBindIndexBuffer(ibuffer, 0.VkDeviceSize, VK_INDEX_TYPE_UINT32)
    cbuffer.vkCmdBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, dsets[i].addr, 0, nil)
    cbuffer.vkCmdDrawIndexed(36u32, 1u32, 0u32, 0i32, 0u32)
    cbuffer.vkCmdEndRenderPass()
    chk cbuffer.vkEndCommandBuffer()

proc newSemaphore(): VkSemaphore =
  let sci = mkVkSemaphoreCreateInfo()
  chk dev.vkCreateSemaphore(sci.unsafeAddr, acs, result.addr)

proc createSemaphores(): array[MAX_FRAMES_IN_FLIGHT,VkSemaphore] =
  for i in 0..<MAX_FRAMES_IN_FLIGHT:
    result[i] = newSemaphore()

proc newFence(): VkFence =
  let fci = mkVkFenceCreateInfo(
    flags = VK_FENCE_CREATE_SIGNALED_BIT.VkFenceCreateFlags,
  )
  chk dev.vkCreateFence(fci.unsafeAddr, acs, result.addr)

proc createFences(): array[MAX_FRAMES_IN_FLIGHT, VkFence] =
  for i in 0..<MAX_FRAMES_IN_FLIGHT:
    result[i] = newFence()

proc updateUniformBuffers(idx: uint32) =
  let view =  mat4f()
  let model = mat4f()
  .translate(pos)
  .rotate(-2.0 * PI * cvMouseY().int.toFloat / cvHeight().int.toFloat, 1, 0, 0)
  .rotate(-2.0 * PI * cvMouseX().int.toFloat / cvWidth().int.toFloat, 0, 1, 0)
  withMappedMemory(umems[idx], ubo, UniformBufferObject):
    ubo[] = UniformBufferObject(
      mvp: proj * view * model,
    )

proc cleanupSwapChain() =
  chk dev.vkDeviceWaitIdle()
  for fbuffer in fbuffers:
    dev.vkDestroyFramebuffer(fbuffer, acs)
  fbuffers.delete(0, fbuffers.high)
  dev.vkFreeCommandBuffers(cpool, cbuffers.len.uint32, cbuffers[0].addr)
  cbuffers.delete(0, cbuffers.high)
  dev.vkDestroyDescriptorPool(dpool, acs)
  dev.vkDestroyPipeline(pline, acs)
  dev.vkDestroyPipelineLayout(layout, acs)
  dev.vkDestroyRenderPass(rpass, acs)
  for iview in iviews:
    dev.vkDestroyImageView(iview, acs)
  iviews.delete(0, iviews.high)
  dev.vkDestroySwapchainKHR(schain, acs)
  for ubuffer in ubuffers:
    dev.vkDestroyBuffer(ubuffer, acs)
  ubuffers.delete(0, ubuffers.high)
  for umem in umems:
    dev.vkFreeMemory(umem, acs)
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
  loadProcs(nil)
  inst = newVkInstance()
  loadProcs(inst)
  dum = newDebugMessenger()
  pdev = inst.pickPhysicalDevice(0)
  surf = newSurface(win, hinst)
  dev = newDevice()
  dev.vkGetDeviceQueue(familyIndex(gfxFamily), 0, gqueue.addr)
  dev.vkGetDeviceQueue(familyIndex(presentFamily), 0, pqueue.addr)
  dslayout = newDescriptorSetLayout()
  vbuffer = newDevBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT.VkBufferUsageFlags, (array[8, Vertex]).sizeOf)
  vmem = allocDevBufferMemory(vbuffer)
  ibuffer = newDevBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT.VkBufferUsageFlags, (array[36, uint32]).sizeOf)
  imem = allocDevBufferMemory(ibuffer)
  available = createSemaphores()
  finished = createSemaphores()
  completed = createFences()
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
  for fence in completed:
    dev.vkDestroyFence(fence, acs)
  for sem in finished:
    dev.vkDestroySemaphore(sem, acs)
  for sem in available:
    dev.vkDestroySemaphore(sem, acs)
  dev.vkDestroyCommandPool(cpool, acs)
  dev.vkFreeMemory(imem, acs)
  dev.vkDestroyBuffer(ibuffer, acs)
  dev.vkFreeMemory(vmem, acs)
  dev.vkDestroyBuffer(vbuffer, acs)
  dev.vkDestroyDescriptorSetLayout(dslayout, acs)
  inst.vkDestroySurfaceKHR(surf, acs)
  dev.vkDestroyDevice(acs)
  inst.vkDestroyDebugUtilsMessengerEXT(dum, acs)
  inst.vkDestroyInstance(acs)


proc draw() =
  let curr = frame mod MAX_FRAMES_IN_FLIGHT
  chk dev.vkWaitForFences(1, completed[curr].addr, VK_TRUE, uint64.high)
  chk dev.vkResetFences(1, completed[curr].addr)
  var idx : uint32
  chk dev.vkAcquireNextImageKHR(schain, uint.high, available[curr], nil.VkFence, idx.addr)
  updateUniformBuffers(idx)
  let ws = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT.VkPipelineStageFlags,]
  let si = mkVkSubmitInfo(
    waitSemaphoreCount = 1,
    pWaitSemaphores = available[curr].addr,
    pWaitDstStageMask = ws[0].unsafeAddr,
    commandBufferCount = 1,
    pCommandBuffers = cbuffers[idx].addr,
    signalSemaphoreCount = 1,
    pSignalSemaphores = finished[curr].addr,
  )
  chk gqueue.vkQueueSubmit(1, si.unsafeAddr, completed[curr])
  let pi = mkVkPresentInfoKHR(
    waitSemaphoreCount = 1,
    pWaitSemaphores = finished[curr].addr,
    swapchainCount = 1,
    pSwapchains = schain.addr,
    pImageIndices = idx.addr,
    )
  chk pqueue.vkQueuePresentKHR(pi.unsafeAddr)
  frame.inc


proc update() =
  if cvPressed cvkLeftArrow: pos.x -= 0.1f
  if cvPressed cvkRightArrow: pos.x += 0.1f
  if cvPressed cvkDownArrow: pos.y += 0.1f
  if cvPressed cvkUpArrow: pos.y -= 0.1f
  if cvPressed cvkX: pos.z -= 0.1f
  if cvPressed cvkZ: pos.z += 0.1f
  draw()

proc resized(w: uint, h: uint) =
  recreateSwapChain()
  proj = perspective(45.0f, w.float / h.float, 0.1f, 100000.0f)

proc handler(e: ptr ev): int {.cdecl.} =
  case evType(e):
    of cveResize:
      resized(e.evWidth(), e.evHeight())
    of cveGLInit:
      initVulkan(cast[pointer](e.evArg0), cast[pointer](e.evArg1))
    of cveGLTerm:
      termVulkan()
    of cveUpdate:
      update()
      return 1
    else:
      return 0

discard cvRun(handler)
{.pop.}
