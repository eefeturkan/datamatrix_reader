import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.datamatrix.decoder.Decoder;

public class DecodePureDM {
  private static BitMatrix toBitMatrix(BufferedImage img) {
    int w = img.getWidth();
    int h = img.getHeight();
    BitMatrix m = new BitMatrix(w, h);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int rgb = img.getRGB(x, y) & 0xFF;
        if (rgb < 128) m.set(x, y);
      }
    }
    return m;
  }
  public static void main(String[] args) throws Exception {
    Decoder dec = new Decoder();
    for (String arg : args) {
      BufferedImage img = ImageIO.read(new File(arg));
      try {
        var res = dec.decode(toBitMatrix(img));
        System.out.println(arg + " => " + res.getText());
      } catch (Throwable t) {
        System.out.println(arg + " => FAIL: " + t.getClass().getSimpleName());
      }
    }
  }
}
