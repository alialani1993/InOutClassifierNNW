using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.Burst;
using UnityEngine;
using UnityEngine.UI;

public class InOutUIController : MonoBehaviour
{
    public GameObject ball;
    private InOutClassifier classifier;
    public TextMeshProUGUI resultsText;
    public TMP_InputField epochsInput;
    // Start is called before the first frame update
    void Start()
    {
        classifier = GetComponent<InOutClassifier>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void Generate()
    {
        ball.transform.position = new Vector2(Random.Range(-7f, 7f), Random.Range(-4f,4f));
    }

    public void Classify()
    {
       
       resultsText.text = classifier.Classify(ball.transform.position.x, ball.transform.position.y);
    }

    public void Train()
    {

        classifier.Train(int.Parse(epochsInput.text));
    }

}
