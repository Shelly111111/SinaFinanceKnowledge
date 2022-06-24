package com.zg.controller;

import com.zg.utils.GraphDBUtil;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

// 和前端对接的接口
@RestController
@CrossOrigin // 跨域的注解
public class ZgController {

    // 这里接收一个参数
    @GetMapping("getDataFromGraphDB")
    public String getDataFromGraphDB(@RequestParam String sparql) throws Exception {
        // keyword 是前端输入后传来的参数 用于拼接在 sparql 语句中
        return GraphDBUtil.Sparql(sparql);
    }

}
