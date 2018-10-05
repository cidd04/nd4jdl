package org.cidd.dlaas;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.HttpURLConnection;

public class Scraper {

  private static final String rapplerUrl = "https://www.rappler.com";

  public static void main(String[] args) {
    System.out.println("GG");

    HttpURLConnection connection = null;
    StringBuilder sb = new StringBuilder();
    try {
//      BufferedReader br = new BufferedReader(new InputStreamReader(connection.getInputStream()));
      String line;
      BufferedReader br = new BufferedReader(new FileReader("D:\\Users\\venr68\\scraper\\src\\main\\resources\\abc.txt"));
      while ((line = br.readLine()) != null) {
        sb.append(line);
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
    String s = sb.toString();
    int startOffset = 0;
//    do {
//      startOffset = loop(s, startOffset);
//    } while (startOffset >= 0);

    //2.
    firstSentence(s, startOffset);

  }

  private static final String thinmargin = "<h4 class=\"thin-margin\">";
  private static final String ahref = "<a href=\"";
  private static final String quote = "\"";
  private static final String link = "\">";
  private static final String endahref = "</a>";

  public static int loop(String s, int startOffset) {
    int thinmarginOffset = s.indexOf(thinmargin, startOffset);
    if (thinmarginOffset < 0) return thinmarginOffset;
    thinmarginOffset += thinmargin.length();
    int ahrefOffsetStart = s.indexOf(ahref, thinmarginOffset) + ahref.length();
    int ahrefOffsetEnd = s.indexOf(quote, ahrefOffsetStart);
    String v = s.substring(ahrefOffsetStart, ahrefOffsetEnd);
    System.out.println(v);
    int titleOffsetStart = s.indexOf(link, ahrefOffsetEnd) + link.length();
    int titleOffsetEnd = s.indexOf(endahref, titleOffsetStart);
    String w = s.substring(titleOffsetStart, titleOffsetEnd);
    System.out.println(w);
    return titleOffsetEnd;
  }

  private static final String ppp = "</p></p>\n<p>";
  private static final String pend = "</p>";

  public static String firstSentence(String s, int startOffset) {
    int ahrefOffsetStart = s.indexOf(ppp, startOffset) + ppp.length();
    int ahrefOffsetEnd = s.indexOf(pend, ahrefOffsetStart);
    String v = s.substring(ahrefOffsetStart, ahrefOffsetEnd);
    System.out.println(v);
    return v;
  }


}
