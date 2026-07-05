case_table={
    'case1':{
        'app_name':['xxxreverse'],
        'app_groups':[ ['xxxreversehost']],
        'fault_start': '2025-05-15 17:33:00',
        'fault_end':'2025-05-15 17:47:00',
        'given_root_cause':'数据库慢sql',
        'hypothesis':"2025-05-15 17:23，平台退款申请(同步)异常，成功率开始下跌，17:33触达P4，经排查业务配置全量退款tips信息，导致慢SQL，同时缓存中无新配置数据，缓存穿透直查数据库，导致负载激增。应急通过业务回滚了配置、数据库紧急扩容、同时调整了数据库主备库的读写比例，在17:47业务指标恢复日常水位，故障恢复。 17:48，业务配置项回滚完成。 17:50，监控恢复至日常水位，故障恢复，请知晓。退款原因TIPS的架构设计不合理，代码架构设计只满足最小使用场景，业务配置全量导致LongSQL&缓存穿透。",
        'suspected_component':'数据库'
    },
    'case2':{
        'app_name':['xtrain'],
        'app_groups':[ ['xtrainhost']],
        'fault_start': '2025-08-10 11:47:00',
        'fault_end':'2025-08-10 11:49:00',
        'given_root_cause':'上游依赖服务不可用',
        'hypothesis':"2025年08月10日11:33，监控告警火车票订单详情页成功率持续下跌，故障原因确认是票务123x6交互接口严重超时，大量大对象进入老年代，导致机器频繁fullgc，于11:48通过扩容&重启等操作故障恢复，根据监控下跌幅度判定，触达P4故障等级。",
        'suspected_component':'上游依赖服务'
    },
    'risk1':{
        'app_name':['xxsearch'],
        'app_groups':[ ['xxsearchhost']],
        'fault_start': '2025-09-26 14:27:20',
        'fault_end':'2025-09-26 14:47:01',
        'given_root_cause':'无实际影响，业务抖动',
        'hypothesis':"无实际影响，业务抖动",
        'suspected_component':'无'
    },
    'risk2':{
        'app_name':['xxx-xxxx-state-machine'],
        'app_groups':[ ['xxx-xxxx-state-machine_xx6xx_host']],
        'fault_start': '2025-10-13 17:07:57',
        'fault_end':'2025-10-13 17:12:06',
        'given_root_cause':'无实际影响，业务抖动',
        'hypothesis':"无实际影响，业务抖动",
        'suspected_component':'无'
    }
}