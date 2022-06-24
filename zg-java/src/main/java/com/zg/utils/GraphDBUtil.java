package com.zg.utils;


import java.io.*;
import java.net.*;
import java.nio.charset.Charset;

// 这里是调用 GraphDB 的工具类，传入一个参数，然后进行查询
public class GraphDBUtil {
    private static String sServerHttp = "http://localhost:7200/";

    // todo 这里填写要查询的仓库名称
    private static String repository="sinafinance";

    // 被调用的查询语句 todo 待加上拼接功能 keyword 就是待拼接的参数 可以为多个参数
    public static String Sparql(String sparql) throws Exception {
        // todo 这句话就是自定义的语句
        String query =
                "PREFIX stockID: <http://www.wust.edu.cn/zg/stockID#>\n" +
                        "PREFIX zg: <http://www.wust.edu.cn/zg#>\n" +
                        "PREFIX sct: <http://www.wust.edu.cn/zg/sct#>\n" +
                        "PREFIX rdfs: <rdfs:subClassOf>\n" + sparql;
        System.out.println(sparql);
        return askQuery(query);
    }

    public static String askQuery(String sQuery){
        try {
            String sUrlEncodedQ ="query="+ URLEncoder.encode(sQuery,"UTF-8");
            String urlStr ="";
            urlStr=sServerHttp+"repositories/"+repository+"?format=tsv&"+ sUrlEncodedQ;
            System.out.println(urlStr);
            String reply = fetchUrl(urlStr);
            return reply;
        } catch (UnsupportedEncodingException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return sQuery;
    }

    public static String fetchUrl(String urlStr) {
        String pageStr = null;
        HttpURLConnection connection = null;
        try {
            URL url = new URL(urlStr);
            if (url.getProtocol().equals("http")||url.getProtocol().equals("https")) {
                CookieHandler.setDefault(new CookieManager(null, CookiePolicy.ACCEPT_ALL));
                // connect to the web server
                connection = (HttpURLConnection) url
                        .openConnection();
                connection.setRequestMethod("GET");
                connection.setRequestProperty("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.155 Safari/537.36");
                connection.setRequestProperty("Accept-Charset", "UTF-8");
                connection.setRequestProperty("Accept", "text/tab-separated-values;q=1.0,application/sparql-results+json;q=0.5,application/rdf+xml;q=0.3");
                //connection.setConnectTimeout(15*1000);
                int responseCode = connection.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    connection.connect();
                    // retrieve the content
                    InputStream stream = connection.getInputStream();
                    pageStr = convertStreamToString(stream);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();

            InputStream stream = connection.getErrorStream();
            if (stream == null) return "Cannot reach server";
            return convertStreamToString(stream);
        }
        return pageStr;
    }
    private static String convertStreamToString(InputStream is) {

        BufferedReader reader = new BufferedReader(new InputStreamReader(is, Charset.forName("utf-8")));
        StringBuilder sb = new StringBuilder();

        String line = null;
        try {
            while ((line = reader.readLine()) != null) {
                sb.append(line + "\n");
            }
        } catch (IOException e) {
            // handle errors ...
        } finally {
            try {
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return sb.toString();
    }

}
