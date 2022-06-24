<template>
    <div id="app">
        <aside class="text-aside">
            新浪财经知识图谱解译平台
        </aside>
        <div style="margin: 20px 0">
            <el-input style="width: 400px" placeholder="请输入关键字" suffix-icon="el-icon-search" v-model="keyword"></el-input>

            <el-button style="margin-left: 10px" class="ml-5" type="primary" @click="getData">搜索</el-button>
            <el-button type="warning" @click="reset">重置</el-button>
        </div>
        <div>
            <div>
                <el-table v-loading="listLoading" :data="list" border fit highlight-current-row style="width: 100%">
                    <el-table-column align="center" v-for="value in formlist" :key="value" :label="value">
                        <template slot-scope="scope">
                            <span>{{ scope.row[value] }}</span>
                        </template>
                    </el-table-column>
                </el-table>
            </div>
        </div>
    </div>
</template>

<script>
    import axios from 'axios'

    export default {
        name: 'App',
        data() {
            return {
                list: null,
                listLoading: false,
                formlist:[],
                keyword: "",
                sparql: "",
            }
        },
        methods: {
            async getData() {
                this.listLoading = true
                axios.post('http://localhost:8000/text2sparql', this.keyword).then(res => {
                    this.sparql = res.data;
                    let url = 'http://localhost:8080/getDataFromGraphDB';
                    axios.get(url, {
                        // params 里面的 keyword 是传给后端要查询的关键字参数
                        params:
                        {
                            sparql: this.sparql,
                        }
                    }).then(res => {
                        // this.content 就是要显示的内容
                        console.log(this.sparql);
                        var str = res.data.split('\n')
                        var title = str[0].split('\t')
                        var len = title.length
                        var data = []
                        for (var i = 1; i < str.length; i++) {
                            var arr = str[i].split('\t')
                            var d = {}
                            for (var j = 0; j < len; j++) {
                                d[title[j]] = arr[j]
                            }
                            data.push(d)
                        }
                        this.formlist = title
                        this.list = data
                    })
                    //console.log(res);
                    this.listLoading = false
                });
            },

            // 这是把输入框清空
            reset() {
                this.keyword = ""
                this.sparql = ""
            }

        }
    }
</script>

<style>
    #app {
        text-align: center;
        margin-top: 60px;
    }
    .text-aside {
        position: relative;
        vertical-align: middle;
        font-size: 30px;
        text-align: center;
        text-size-adjust: auto;
    }
</style>
